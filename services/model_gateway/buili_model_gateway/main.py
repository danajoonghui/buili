from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from services.api.buili.gpu import force_gpu_7, gpu_policy

force_gpu_7()

app = FastAPI(title="Buili Model Gateway", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str = "buili-local-reasoner"
    messages: list[ChatMessage]
    temperature: float = 0.1


class EmbeddingRequest(BaseModel):
    model: str = "buili-hash-embedding"
    input: str | list[str]


def _hash_embedding(text: str, dims: int = 384) -> list[float]:
    vec = np.zeros(dims, dtype=np.float32)
    for token in text.lower().split():
        idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dims
        vec[idx] += 1
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return [float(v) for v in vec]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", **gpu_policy()}


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatRequest) -> dict[str, Any]:
    prompt = "\n".join(str(message.content) for message in payload.messages)
    content = {
        "issue_type": "unverified",
        "confidence": 0.52,
        "recommended_action": (
            "Require cited plan/spec evidence and human PM review before approval."
        ),
        "model_profile": payload.model,
        "input_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    }
    return {
        "id": f"chatcmpl-{content['input_hash']}",
        "object": "chat.completion",
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(payload: EmbeddingRequest) -> dict[str, Any]:
    texts = payload.input if isinstance(payload.input, list) else [payload.input]
    return {
        "object": "list",
        "model": payload.model,
        "data": [
            {"object": "embedding", "index": idx, "embedding": _hash_embedding(text)}
            for idx, text in enumerate(texts)
        ],
    }


if __name__ == "__main__":
    uvicorn.run("services.model_gateway.buili_model_gateway.main:app", host="0.0.0.0", port=8100)
