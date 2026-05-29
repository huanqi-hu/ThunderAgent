"""Synthetic multi-turn agent workload for benchmarking ThunderAgent.

Each "program" runs N turns:
  send chat -> wait response -> sleep a tool_time -> send next turn
Context grows turn by turn (we append the assistant reply + a new user turn),
so KV-cache pressure goes up exactly like a real agent.

Args:
  --target      base URL (with or without /v1)
  --programs    number of parallel agentic programs
  --turns       steps per program
  --tool-mean   mean tool-call time (gamma distributed)
  --use-program-id  if set, add extra_body.program_id (route through ThunderAgent)
"""
import argparse
import asyncio
import json
import random
import time
import uuid
import os
import statistics
from typing import List

import aiohttp


def gen_prompt(n_chars: int) -> str:
    return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * max(1, n_chars // 60)


async def one_program(session: aiohttp.ClientSession, base_url: str, turns: int,
                       prompt_chars: int, tool_mean: float, use_pid: bool, out: list, idx: int):
    pid = f"prog-{idx}-{uuid.uuid4().hex[:6]}"
    messages = [
        {"role": "system", "content": "You are a benchmark agent."},
        {"role": "user", "content": gen_prompt(prompt_chars)},
    ]
    chat_url = f"{base_url}/v1/chat/completions"
    release_url = f"{base_url}/programs/release"
    t0 = time.time()
    latencies = []
    for step in range(turns):
        payload = {
            "model": "mock",
            "messages": messages,
            "stream": False,
            "max_completion_tokens": 64,
        }
        if use_pid:
            payload["extra_body"] = {"program_id": pid}
        t_send = time.time()
        try:
            async with session.post(chat_url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as r:
                resp = await r.json()
        except Exception as e:
            out.append({"pid": pid, "error": str(e), "duration": time.time() - t0})
            return
        latencies.append(time.time() - t_send)
        assistant = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": f"step {step+1}: {gen_prompt(int(prompt_chars * 0.2))}"})
        if tool_mean > 0:
            await asyncio.sleep(random.gammavariate(2.0, tool_mean / 2.0))

    if use_pid:
        try:
            async with session.post(release_url, json={"program_id": pid},
                                    timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception:
            pass

    out.append({
        "pid": pid, "duration": time.time() - t0,
        "n_steps": turns,
        "p50_step_s": statistics.median(latencies),
        "p95_step_s": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        "max_step_s": max(latencies),
    })


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--programs", type=int, default=16)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--prompt-chars", type=int, default=400)
    p.add_argument("--tool-mean", type=float, default=0.2)
    p.add_argument("--use-program-id", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()

    base_url = args.target.rstrip("/")
    out: list = []
    conn = aiohttp.TCPConnector(limit=args.programs * 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        t_start = time.time()
        tasks = [
            asyncio.create_task(one_program(
                session, base_url, args.turns, args.prompt_chars,
                args.tool_mean, args.use_program_id, out, i))
            for i in range(args.programs)
        ]
        await asyncio.gather(*tasks)
        wall = time.time() - t_start

    ok = [r for r in out if "error" not in r]
    bad = [r for r in out if "error" in r]
    total_steps = sum(r["n_steps"] for r in ok)
    summary = {
        "target": args.target,
        "use_program_id": args.use_program_id,
        "programs": args.programs,
        "turns_per_program": args.turns,
        "wall_clock_s": round(wall, 3),
        "ok": len(ok),
        "errors": len(bad),
        "total_steps": total_steps,
        "throughput_steps_per_min": round(total_steps / wall * 60, 2) if wall > 0 else 0,
        "throughput_programs_per_min": round(len(ok) / wall * 60, 2) if wall > 0 else 0,
        "avg_program_duration_s": round(sum(r["duration"] for r in ok) / max(1, len(ok)), 3),
        "p50_step_s": round(statistics.median([r["p50_step_s"] for r in ok]), 3) if ok else None,
        "p95_step_s": round(max(r["p95_step_s"] for r in ok), 3) if ok else None,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": out}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
