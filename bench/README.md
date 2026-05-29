# ThunderAgent 本地性能测试套件

环境已用 `uv` 在 `/home/huanqi/projects/ThunderAgent/.venv` 里准备好。所有命令都在仓库根目录执行。

## 0. 激活环境

```bash
cd /home/huanqi/projects/ThunderAgent
source .venv/bin/activate

# 本机有 socks/http 代理，会把 httpx 卡住，所有终端先 unset 一下
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

已装：`thunderagent==0.1.0` (editable), `fastapi`, `uvicorn`, `httpx`, `aiohttp`。

## 1. 用 Mock vLLM 验证调度逻辑（**0 GPU 也能跑**）

ThunderAgent 论文里要测的是 *program-aware scheduler* 的效果，所以哪怕没有真模型，只要后端的 `/metrics` 和 `/v1/chat/completions` 接口语义对，就能验证 pause/resume/排队/BFD 装箱是否生效。

```bash
# 终端 A: 起 mock vLLM（CPU only）
source .venv/bin/activate
python bench/mock_vllm.py --port 18000 --num-gpu-blocks 256 --resp-tokens 60 \
                          --prefill-s 0.05 --per-token-s 0.002

# 终端 B: 起 ThunderAgent (TR 模式 = paper 的核心调度)
source .venv/bin/activate
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
thunderagent --backend-type vllm --backends http://localhost:18000 \
             --port 19000 --router tr \
             --metrics --metrics-interval 1.0 \
             --scheduler-interval 1.0 \
             --acting-token-weight 1.0 --use-acting-token-decay \
             --profile --profile-dir /tmp/thunderagent_profiles

# 终端 C: 跑两组对比 — 直连 mock vs. 走 ThunderAgent

# (1) baseline：直连 mock vLLM，不带 program_id
python bench/agent_workload.py --target http://localhost:18000 \
       --programs 32 --turns 4 --tool-mean 0.2 --out /tmp/bench_direct.json

# (2) 走 ThunderAgent，开启 program_id 调度
python bench/agent_workload.py --target http://localhost:19000 \
       --programs 32 --turns 4 --tool-mean 0.2 \
       --use-program-id --out /tmp/bench_thunder.json

# 观察 ThunderAgent 的实时状态
curl -s http://localhost:19000/health    | python -m json.tool
curl -s http://localhost:19000/metrics   | python -m json.tool | head -40
curl -s http://localhost:19000/profiles  | python -m json.tool | head -40

# Per-step CSV
cat /tmp/thunderagent_profiles/step_profiles.csv | head -20
```

**期望现象**：
- 当 `--num-gpu-blocks` 调小（容量小），baseline 因 thrashing 会出现长尾延迟；
- 走 ThunderAgent 的 `pause_s` 字段会非 0（有程序被排队），但单步 `prefill_s`/`decode_s` 更稳定；
- `/health` 里 `reasoning_count + acting_count + paused_count = programs`；
- `/metrics` 的 `active_program_tokens` ≈ vLLM 报告的 `kv_cache_usage_perc × capacity`，差额就是 `shared_tokens`。

## 2. 跑真 vLLM（卡空了之后）

```bash
# 1) 装 vllm（torch 自动匹配）
uv pip install vllm --torch-backend=auto

# 2) 起一个小模型（RTX 5090 32GB 单卡，Qwen3-8B FP16 大约要 ~16GB KV cache）
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
    --port 18000 --max-model-len 8192 --gpu-memory-utilization 0.55

# 3) ThunderAgent（同 §1，命令完全一样）

# 4) 用例子跑 — 比如 mini-SWE-agent on SWE-Bench Lite
#    见 examples/inference/mini-swe-agent/scripts/reproduce/  里的 reproduce_*.sh
```

`--gpu-memory-utilization` 不要给太大，因为另一张 5090 现在也有人在用；先留 5GB headroom。

如果想跑 SGLang 版本，把 `--backend-type sglang` 换上，ThunderAgent 会去 `GET /get_server_info` 拿 `max_total_num_tokens`（见 `ThunderAgent/backend/sglang_metrics.py:199-228`）。

## 3. 一些已知陷阱

| 问题 | 处理 |
|---|---|
| `socksio` ImportError | 启动前 `unset ALL_PROXY all_proxy` —— 本机环境有 socks5 代理，httpx ≥0.28 会要求专门安装 `httpx[socks]`。ThunderAgent 自己不需要走代理，unset 即可 |
| `cache_config=None` 警告 | 后端 `/metrics` 还没返回 `vllm:cache_config_info`，等 vLLM 完全启动（通常 10-30s） |
| 客户端 timeout | `agent_workload.py` 里默认 120s；真模型大请求时改 600 |
| Profile CSV 不写 | 必须加 `--profile`；CSV 路径在 `--profile-dir` |
| 看到 `pause_s = 0` | 当前并发还没把后端打到 capacity overflow；增大 `--programs` 或减小 `--num-gpu-blocks` |

## 4. 文件说明

- `bench/mock_vllm.py` — 200 行 FastAPI 假后端，导出 Prometheus 格式 metrics + OpenAI chat completions（流/非流都支持），可配 capacity 触发 thrashing。
- `bench/agent_workload.py` — async 多程序压测客户端。每个 program N 轮 chat + 随机 gamma 分布 tool-time，结束时调 `/programs/release`。

## 5. 想看的话 — 这俩 endpoint 在 paper 里对应什么

- `/health`、`/programs` — Table 3a 的 `ProgramState`（status/backend_url/step_count/total_tokens）。
- `/metrics`（注意是 **ThunderAgent** 的 :19000/metrics，不是 vLLM 的 :18000/metrics） — Table 4 的 `BackendState`（url/healthy/cache_config/active_program_tokens）。
- `/profiles/{program_id}` — Figure 6a 的 4 段时间拆解：prefill/decode/pause/tool-call。
