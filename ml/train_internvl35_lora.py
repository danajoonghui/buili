from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.gpu import assert_gpu_7, force_gpu_7, gpu_policy

force_gpu_7()

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-14B-HF"
TEACHER_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(row: dict[str, Any], *, include_answer: bool) -> list[dict[str, Any]]:
    user_content = [
        {"type": "image", "image": str(Path(image_path).resolve())} for image_path in row["images"]
    ]
    user_content.append({"type": "text", "text": row["prompt"]})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def tokenize_example(processor: Any, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    prompt_inputs = processor.apply_chat_template(
        build_messages(row, include_answer=False),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_inputs = processor.apply_chat_template(
        build_messages(row, include_answer=True),
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    labels = full_inputs["input_ids"].clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    full_inputs["labels"] = labels
    return full_inputs


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def json_valid_rate(samples: list[str]) -> float:
    if not samples:
        return 0.0
    valid = 0
    for text in samples:
        try:
            json.loads(text[text.find("{") : text.rfind("}") + 1])
            valid += 1
        except Exception:
            pass
    return valid / len(samples)


def generate_eval_sample(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
    *,
    max_new_tokens: int,
) -> str:
    inputs = processor.apply_chat_template(
        build_messages(row, include_answer=False),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_to_device(inputs, device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.decode(new_tokens[0], skip_special_tokens=True).strip()


def target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("At least one LoRA target module is required.")
    return modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--teacher-model-id", default=TEACHER_MODEL_ID)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/buili_vlm/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/artifacts/buili_internvl35_14b_lora"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=96)
    parser.add_argument("--max-eval-samples", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", default="")
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-generation-eval", action="store_true")
    parser.add_argument("--quantization", choices=["4bit", "bf16"], default="4bit")
    args = parser.parse_args()

    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train outside GPU 7.")
    device = torch.device("cuda:0")
    random.seed(42)
    torch.manual_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset)
    dataset_manifest = read_optional_json(args.dataset.parent / "dataset_manifest.json")
    open_corpus_manifest = read_optional_json(
        REPO_ROOT / "data" / "processed" / "open_construction_corpus" / "manifest.json"
    )
    train_rows = [row for row in rows if row.get("split") == "train"][: args.max_train_samples]
    eval_rows = [row for row in rows if row.get("split") == "eval"][: args.max_eval_samples]
    if not train_rows:
        raise RuntimeError("No training rows found.")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)

    lora_layers = [int(value) for value in args.lora_layers.split(",") if value.strip()]
    lora_kwargs: dict[str, Any] = {}
    if lora_layers:
        lora_kwargs = {"layers_to_transform": lora_layers, "layers_pattern": "layers"}
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules(args.target_modules),
        **lora_kwargs,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    planned_rows = len(train_rows) * args.epochs
    if args.max_steps > 0:
        planned_rows = min(planned_rows, args.max_steps * args.gradient_accumulation_steps)
    total_updates = max(1, math.ceil(planned_rows / args.gradient_accumulation_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_updates // 10),
        num_training_steps=total_updates,
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    seen_rows = 0
    running_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        progress = tqdm(train_rows, desc=f"epoch {epoch}/{args.epochs}")
        for row_idx, row in enumerate(progress, start=1):
            batch = move_to_device(tokenize_example(processor, row), device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            seen_rows += 1
            should_step = (
                seen_rows % args.gradient_accumulation_steps == 0
                or row_idx == len(train_rows)
                or (args.max_steps > 0 and global_step + 1 >= args.max_steps)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                mean_loss = running_loss / max(args.gradient_accumulation_steps, 1)
                history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": mean_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
                progress.set_postfix({"loss": f"{mean_loss:.4f}", "step": global_step})
                running_loss = 0.0
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    eval_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for row in tqdm(eval_rows, desc="eval_loss"):
            batch = move_to_device(tokenize_example(processor, row), device)
            eval_losses.append(float(model(**batch).loss.detach().cpu()))

    generations: list[dict[str, str]] = []
    if eval_rows and not args.skip_generation_eval:
        for row in tqdm(eval_rows[: min(4, len(eval_rows))], desc="eval_generate"):
            generated = generate_eval_sample(
                model,
                processor,
                row,
                device,
                max_new_tokens=args.max_new_tokens,
            )
            generations.append(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "expected": row["answer"],
                    "generated": generated,
                }
            )

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir / "processor")
    summary = {
        "status": "trained",
        "model_family": "InternVL3.5",
        "base_model_id": args.model_id,
        "teacher_model_id": args.teacher_model_id,
        "adapter_path": str(args.out_dir),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "quantization": args.quantization,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "target_modules": sorted(lora_config.target_modules),
            "layers_to_transform": lora_layers or "all",
        },
        "dataset": str(args.dataset),
        "dataset_manifest": dataset_manifest,
        "open_corpus_manifest": open_corpus_manifest,
        "data_governance": {
            "commercial_training_claim": (
                "Only public/government/open-license rows collected into the local manifest "
                "are included. Gated or CC-BY-NC datasets are excluded unless separately "
                "licensed."
            ),
            "required_next_stage": (
                "Replace or augment weak public labels with licensed customer field photos, "
                "plan PDFs, inspection outcomes, and PM acceptance labels before production "
                "accuracy claims."
            ),
        },
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "epochs": args.epochs,
        "global_steps": global_step,
        "train_history": history,
        "eval_loss": sum(eval_losses) / len(eval_losses) if eval_losses else None,
        "json_valid_rate": json_valid_rate([item["generated"] for item in generations]),
        "generation_eval": generations,
        "silicon_valley_output_contract": {
            "required_sections": [
                "executive_summary",
                "requirement",
                "observation",
                "evidence_chain",
                "risk",
                "recommended_action",
                "rfi_draft",
                "punch_list_row",
                "change_order_evidence",
                "decision_gate",
            ],
            "business_standard": (
                "Outputs must be PM-review-ready issue packages, not loose captions. "
                "Every candidate needs plan citation, field evidence status, risk, next "
                "action, RFI/punch/CO routing, and human verification gate."
            ),
        },
        "trained_at": int(time.time()),
        "scope_note": (
            "This is a real 14B VLM LoRA domain-adaptation run on public Buili drawing and "
            "field-evidence data. Production readiness still requires pilot customer data, "
            "larger labeled field media, and external QA before market claims."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    adapter_manifest = {
        "status": summary["status"],
        "model_family": summary["model_family"],
        "base_model_id": summary["base_model_id"],
        "teacher_model_id": summary["teacher_model_id"],
        "adapter_path": summary["adapter_path"],
        "adapter_files": sorted(path.name for path in args.out_dir.glob("adapter*")),
        "dataset": summary["dataset"],
        "dataset_sha256": (dataset_manifest or {}).get("sha256"),
        "open_corpus_version": (open_corpus_manifest or {}).get("corpus_version"),
        "train_rows": summary["train_rows"],
        "eval_rows": summary["eval_rows"],
        "global_steps": summary["global_steps"],
        "eval_loss": summary["eval_loss"],
        "json_valid_rate": summary["json_valid_rate"],
        "gpu": summary["gpu"],
        "trained_at": summary["trained_at"],
    }
    (args.out_dir / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
