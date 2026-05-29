"""Mock vLLM backend for ThunderAgent benchmarking.

Exposes:
  GET  /metrics                    Prometheus-format metrics ThunderAgent parses
  POST /v1/chat/completions        Streaming + non-streaming, returns synthetic usage

It emulates:
- A fixed KV-cache capacity (configurable via env CAP_TOKENS, default 8192).
- Per-program prefix cache reuse so kv_cache_usage_perc < sum(tokens)/cap.
- A small per-request latency so we can observe queueing.
"""
import argparse
import asyncio
import json
import random
import time
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse

app = FastAPI()

# ---- configurable via CLI ----
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 512                 # ⇒ capacity = 8192 tokens by default
PER_REQUEST_BASE_S = 0.15            # prefill time floor
PER_TOKEN_S = 0.003                  # decode "speed"
RESP_TOKENS = 80                     # tokens emitted per response
THRASH_PENALTY = 0.0                 # extra seconds per token over-capacity (simulates KV thrash recompute)

# ---- state ----
state = {
    "active_tokens": 0,
    "prefix_hits": 0,
    "prefix_queries": 0,
    "prompt_tokens_total": 0,
    "gen_tokens_total": 0,
    "running": 0,
    "preemptions": 0,
}
lock = asyncio.Lock()


@app.post("/_reset")
async def reset():
    async with lock:
        for k in state:
            state[k] = 0
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    """Prometheus text output matching what ThunderAgent vllm_metrics.py parses."""
    cap = BLOCK_SIZE * NUM_GPU_BLOCKS
    usage = min(1.0, state["active_tokens"] / cap) if cap else 0.0
    body = []
    body.append(f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",num_gpu_blocks="{NUM_GPU_BLOCKS}",model="mock"}} 1.0')
    body.append(f'vllm:num_requests_running{{model="mock"}} {state["running"]}')
    body.append(f'vllm:num_requests_waiting{{model="mock"}} 0')
    body.append(f'vllm:kv_cache_usage_perc{{model="mock"}} {usage:.6f}')
    body.append(f'vllm:prefix_cache_queries_total{{model="mock"}} {state["prefix_queries"]}')
    body.append(f'vllm:prefix_cache_hits_total{{model="mock"}} {state["prefix_hits"]}')
    body.append(f'vllm:prompt_tokens_total{{model="mock"}} {state["prompt_tokens_total"]}')
    body.append(f'vllm:generation_tokens_total{{model="mock"}} {state["gen_tokens_total"]}')
    body.append(f'vllm:num_preemptions_total{{model="mock"}} {state["preemptions"]}')
    body.append(f'vllm:request_success_total{{model="mock",finished_reason="stop"}} 0')
    return PlainTextResponse("\n".join(body) + "\n")


def _count_chars(messages):
    n = 0
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for it in c:
                if isinstance(it, dict):
                    t = it.get("text") or it.get("input_text") or ""
                    n += len(t)
    return n


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    char_count = _count_chars(messages)
    # 1 token ≈ 4 chars (close to vLLM tokenizer behavior)
    prompt_tokens = max(1, char_count // 4)
    # Simulate prefix cache: 30% of tokens are cached when prompt is short, ramp up with length
    cached = int(prompt_tokens * (0.2 + 0.5 * min(1.0, prompt_tokens / 1000)))
    completion_tokens = RESP_TOKENS

    async with lock:
        state["active_tokens"] += prompt_tokens + completion_tokens
        state["prefix_queries"] += prompt_tokens
        state["prefix_hits"] += cached
        state["prompt_tokens_total"] += prompt_tokens
        state["gen_tokens_total"] += completion_tokens
        state["running"] += 1
    try:
        # Simulate prefill latency proportional to *uncached* prompt
        prefill_s = PER_REQUEST_BASE_S + (prompt_tokens - cached) * 0.00005
        # Simulate KV cache thrashing: when total tokens > capacity, all
        # requests pay a recompute penalty proportional to the overflow.
        cap = BLOCK_SIZE * NUM_GPU_BLOCKS
        if THRASH_PENALTY > 0 and state["active_tokens"] > cap:
            overflow = state["active_tokens"] - cap
            prefill_s += overflow * THRASH_PENALTY
        await asyncio.sleep(prefill_s)

        stream = body.get("stream", False)
        if stream:
            return StreamingResponse(_stream(prompt_tokens, completion_tokens, cached),
                                     media_type="text/event-stream")
        # Non-streaming
        await asyncio.sleep(completion_tokens * PER_TOKEN_S)
        return JSONResponse({
            "id": f"chatcmpl-mock-{random.randint(0,1<<32):x}",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        })
    finally:
        async with lock:
            # Release "decode" capacity but keep prefix in KV (simulate)
            state["active_tokens"] = max(0, state["active_tokens"] - completion_tokens)
            state["running"] -= 1


async def _stream(prompt_tokens, completion_tokens, cached):
    base = {
        "id": f"chatcmpl-mock-{random.randint(0,1<<32):x}",
        "object": "chat.completion.chunk",
        "model": "mock",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(base)}\n\n".encode()
    for i in range(completion_tokens):
        await asyncio.sleep(PER_TOKEN_S)
        chunk = {
            "id": base["id"], "object": "chat.completion.chunk", "model": "mock",
            "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    last = {
        "id": base["id"], "object": "chat.completion.chunk", "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    yield f"data: {json.dumps(last)}\n\n".encode()
    yield b"data: [DONE]\n\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--num-gpu-blocks", type=int, default=NUM_GPU_BLOCKS)
    p.add_argument("--prefill-s", type=float, default=PER_REQUEST_BASE_S)
    p.add_argument("--per-token-s", type=float, default=PER_TOKEN_S)
    p.add_argument("--resp-tokens", type=int, default=RESP_TOKENS)
    p.add_argument("--thrash-penalty", type=float, default=THRASH_PENALTY,
                   help="seconds added per over-capacity token (simulates KV thrash recompute)")
    args = p.parse_args()
    BLOCK_SIZE = args.block_size
    NUM_GPU_BLOCKS = args.num_gpu_blocks
    PER_REQUEST_BASE_S = args.prefill_s
    PER_TOKEN_S = args.per_token_s
    RESP_TOKENS = args.resp_tokens
    THRASH_PENALTY = args.thrash_penalty
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
