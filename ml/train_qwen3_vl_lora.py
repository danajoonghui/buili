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
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
PRODUCTION_TARGET_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct-FP8"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_messages(
    row: dict[str, Any],
    *,
    include_answer: bool,
    add_generation_prompt: bool,
) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": str(Path(path).resolve())} for path in row["images"]]
    content.append({"type": "text", "text": row["prompt"]})
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": row["answer"]})
    return messages


def tokenize_example(processor: Any, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    prompt_messages = build_messages(row, include_answer=False, add_generation_prompt=True)
    full_messages = build_messages(row, include_answer=True, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
    )
    labels = full_inputs["input_ids"].clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    for token in ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            labels[labels == token_id] = -100
    full_inputs["labels"] = labels
    return full_inputs


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


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
    messages = build_messages(row, include_answer=False, add_generation_prompt=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
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
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--production-target-model-id", default=PRODUCTION_TARGET_MODEL_ID)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/buili_vlm/sft_dataset.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/buili_qwen3_vl_lora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=120)
    parser.add_argument("--max-eval-samples", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--lora-layers",
        default="",
        help="Comma-separated transformer layer indices for PEFT layers_to_transform.",
    )
    parser.add_argument("--attention-only-lora", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quantization",
        choices=["8bit_offload", "4bit", "bf16"],
        default="8bit_offload",
    )
    parser.add_argument("--max-gpu-memory", default="30GiB")
    parser.add_argument("--max-cpu-memory", default="240GiB")
    parser.add_argument("--skip-generation-eval", action="store_true")
    args = parser.parse_args()

    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train outside GPU 7.")
    device = torch.device("cuda:0")
    random.seed(42)
    torch.manual_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset)
    train_rows = [row for row in rows if row.get("split") == "train"][: args.max_train_samples]
    eval_rows = [row for row in rows if row.get("split") == "eval"][: args.max_eval_samples]
    if not train_rows:
        raise RuntimeError("No training rows found.")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    quantization_config = None
    device_map: str | dict[str, int] = {"": 0}
    max_memory = None
    offload_folder = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit_offload":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        device_map = "auto"
        max_memory = {0: args.max_gpu_memory, "cpu": args.max_cpu_memory}
        offload_folder = str(args.out_dir / "offload")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    elif args.quantization == "8bit_offload" and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not args.attention_only_lora:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    lora_layers = [int(value) for value in args.lora_layers.split(",") if value.strip()]
    lora_kwargs: dict[str, Any] = {}
    if lora_layers:
        lora_kwargs = {"layers_to_transform": lora_layers, "layers_pattern": "layers"}
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        **lora_kwargs,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    total_updates = max(
        1,
        math.ceil(len(train_rows) * args.epochs / args.gradient_accumulation_steps),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_updates // 10),
        num_training_steps=total_updates,
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    running_loss = 0.0
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        progress = tqdm(train_rows, desc=f"epoch {epoch}/{args.epochs}")
        for row_idx, row in enumerate(progress, start=1):
            batch = tokenize_example(processor, row)
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            should_step = row_idx % args.gradient_accumulation_steps == 0 or row_idx == len(
                train_rows
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

    eval_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for row in tqdm(eval_rows, desc="eval_loss"):
            batch = tokenize_example(processor, row)
            batch = move_to_device(batch, device)
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
        "model_family": "Qwen3-VL",
        "base_model_id": args.model_id,
        "production_serving_target_model_id": args.production_target_model_id,
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
            "product_positioning": (
                "MEP/electrical field-to-report review layer for PMs, supers, and trade "
                "partners; candidate generation only, with human verification gates."
            ),
        },
        "adapter_path": str(args.out_dir),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "quantization": args.quantization,
        "max_gpu_memory": args.max_gpu_memory,
        "max_cpu_memory": args.max_cpu_memory,
        "device_map_sample": list(getattr(model, "hf_device_map", {}).items())[:32],
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "target_modules": sorted(lora_config.target_modules),
            "layers_to_transform": lora_layers or "all",
        },
        "dataset": str(args.dataset),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "epochs": args.epochs,
        "global_steps": global_step,
        "train_history": history,
        "eval_loss": sum(eval_losses) / len(eval_losses) if eval_losses else None,
        "json_valid_rate": json_valid_rate([item["generated"] for item in generations]),
        "generation_eval": generations,
        "trained_at": int(time.time()),
        "scope_note": (
            "This is a real Qwen3-VL LoRA domain-adaptation run for Buili public "
            "drawing/field-evidence workflows. Production claims still require pilot "
            "data, larger labeled field sets, and customer acceptance testing."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
