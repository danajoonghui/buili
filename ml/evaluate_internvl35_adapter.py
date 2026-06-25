from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.buili.gpu import assert_gpu_7, force_gpu_7, gpu_policy

force_gpu_7()

from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-14B-HF"
PRODUCTION_SUFFIX = (
    "\n\nProduction output constraint: return compact valid JSON only. "
    "Do not use markdown fences. Close every array and object. For issue, plan, and "
    "report tasks include PM-ready fields such as executive_summary, requirement, "
    "observation, evidence_chain, risk, recommended_action, rfi_draft, punch_list_row, "
    "change_order_evidence, and decision_gate when applicable. For field evidence "
    "tasks include evidence_limitations, pm_gate, conclusions, and issues[] with "
    "requirement, evidence, plan_location, risk, and next_action. Do not make a final "
    "defect decision automatically."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_messages(row: dict[str, Any], *, prompt_suffix: str) -> list[dict[str, Any]]:
    user_content = [
        {"type": "image", "image": str(Path(image_path).resolve())} for image_path in row["images"]
    ]
    user_content.append({"type": "text", "text": row["prompt"] + prompt_suffix})
    return [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {"role": "user", "content": user_content},
    ]


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def contract_hits(parsed: dict[str, Any] | None) -> list[str]:
    if not parsed:
        return []
    expected = [
        "executive_summary",
        "requirement",
        "observation",
        "observations",
        "evidence_chain",
        "risk",
        "recommended_action",
        "rfi_draft",
        "punch_list_row",
        "change_order_evidence",
        "decision_gate",
        "human_review_required",
        "final_defect_decision",
        "evidence_limitations",
        "pm_gate",
        "conclusions",
        "issues",
    ]
    hits = [key for key in expected if key in parsed]
    issues = parsed.get("issues")
    if isinstance(issues, list) and issues:
        nested_expected = ["requirement", "evidence", "plan_location", "risk", "next_action"]
        for key in nested_expected:
            if any(isinstance(issue, dict) and key in issue for issue in issues):
                hits.append(f"issues[].{key}")
    return hits


def select_eval_rows(rows: list[dict[str, Any]], max_eval_samples: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task") or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    task_names = sorted(by_task)
    while len(selected) < max_eval_samples and any(by_task.values()):
        for task in task_names:
            bucket = by_task[task]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= max_eval_samples:
                    break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("data/artifacts/buili_internvl35_14b_lora"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/buili_vlm/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/artifacts/buili_internvl35_14b_lora/generation_qa.json"),
    )
    parser.add_argument("--max-eval-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--prompt-suffix", default=PRODUCTION_SUFFIX)
    args = parser.parse_args()

    assert_gpu_7()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to evaluate outside GPU 7.")
    device = torch.device("cuda:0")
    rows = [row for row in read_jsonl(args.dataset) if row.get("split") == "eval"]
    rows = select_eval_rows(rows, args.max_eval_samples)

    processor_path = args.adapter_dir / "processor"
    processor = AutoProcessor.from_pretrained(
        processor_path if processor_path.exists() else args.model_id,
        trust_remote_code=True,
    )
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
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    results: list[dict[str, Any]] = []
    for row in rows:
        inputs = processor.apply_chat_template(
            build_messages(row, prompt_suffix=args.prompt_suffix),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_to_device(inputs, device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.decode(new_tokens[0], skip_special_tokens=True).strip()
        parsed = extract_json(text)
        hits = contract_hits(parsed)
        results.append(
            {
                "id": row["id"],
                "task": row["task"],
                "json_valid": parsed is not None,
                "contract_hits": hits,
                "contract_hit_count": len(hits),
                "generated": text,
            }
        )

    json_valid_rate = (
        sum(1 for item in results if item["json_valid"]) / len(results) if results else 0.0
    )
    avg_contract_hit_count = (
        sum(item["contract_hit_count"] for item in results) / len(results) if results else 0.0
    )
    workflow_contract_rate = (
        sum(1 for item in results if item["contract_hit_count"] >= 3) / len(results)
        if results
        else 0.0
    )
    report = {
        "status": "evaluated",
        "model_family": "InternVL3.5",
        "base_model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir),
        "gpu": gpu_policy(),
        "torch_device": torch.cuda.get_device_name(0),
        "max_eval_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "json_valid_rate": json_valid_rate,
        "avg_contract_hit_count": avg_contract_hit_count,
        "workflow_contract_rate": workflow_contract_rate,
        "evaluated_at": int(time.time()),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
