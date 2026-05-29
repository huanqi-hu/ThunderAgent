# ThunderAgent 代码深度解析 (Code Walkthrough)

> 目标：把论文 *ThunderAgent: A Fast, Simple, and Program-Aware Agentic Inference System*（`assets/paper/_Arxiv__ThunderAgent.pdf`）中提到的每个技术点对应到仓库里**实际的代码位置**，并解释作者如何（以及在多大程度上）实现了 paper 里的设计。

---

## 1. TL;DR — 这个仓库到底实现了什么？

ThunderAgent 仓库本身只包含一个 **~1500 行的 Python 中间件**（`ThunderAgent/` 目录），它实现的是论文里 *"program-aware scheduling layer"* 的部分；其余庞大的 `examples/` 目录是**完全独立**的下游项目（OpenHands / mini-SWE-agent / ToolOrchestra / SkyRL / slime / Harbor），用来在论文里跑 benchmark。

具体来说，ThunderAgent 本体实现了：

| 论文章节 | 论文术语 | 仓库对应位置 |
|---|---|---|
| §4.1 | Program Abstraction `P=⟨ID, c, T, L, τ, s⟩` | [ThunderAgent/program/state.py](ThunderAgent/program/state.py)（`Program` dataclass） |
| §4.2 | Cost Model（STP） | 隐式：`backend.active_program_tokens`、`shared_tokens` 等是 STP 的瞬时积分项 |
| §4.3.1 | Periodic thrashing detection + Pause/Restore | [ThunderAgent/scheduler/router.py:660-718](ThunderAgent/scheduler/router.py#L660)（`_scheduled_check`, `_pause_until_safe`） |
| §4.3.1 | Shortest-First Eviction | [router.py:519-543, 685-714](ThunderAgent/scheduler/router.py#L519)（`_get_acting_programs_sorted(ascending=True)`） |
| §4.3.1 | Time-decay function f(t)=2⁻ᵗ | [backend/state.py:196-214](ThunderAgent/backend/state.py#L196)（`remaining_capacity_with_decay`） |
| §4.3.2 | Global program-aware waiting queue | [router.py:92-93, 719-844](ThunderAgent/scheduler/router.py#L92)（`global_waiting_queue` + BFD 装箱） |
| §4.4 | Hook-based garbage collector | [app.py:137-151](ThunderAgent/app.py#L137)（`/programs/release` 端点） |
| §4.4 | Asynchronous environment preparation | **未在 ThunderAgent 内部实现**；通过 `program_id` 提前 register 让 agent 客户端自己异步起 Docker（设计上的 hook，agent 侧自己 schedule） |
| Appendix B | OpenAI-compatible passthrough + program_id | [scheduler/vllm_request_processor.py:82-94](ThunderAgent/scheduler/vllm_request_processor.py#L82)（`remove_program_id` 在转发前剥掉） |

不在仓库内的：
- 不修改 vLLM、SGLang 任何源码（**完全 0 侵入**，下面会专门讨论）。
- 不内置 sandbox 实现 —— 论文里所谓的 "sandbox 生命周期管理" 在开源代码里**只是一个 HTTP hook**，真正的 docker 启停由调用 ThunderAgent 的 agent 自己做。
- 没有 multi-tier KV cache offloading、PD disaggregation 之类的内核级修改。

---

## 2. 顶层数据流（从一个 chat completion 请求看起）

入口在 `ThunderAgent/__main__.py:6-78`：解析 CLI → 写 global Config → uvicorn 起 FastAPI。

```
┌─────── Client ────────┐
│ openai.chat.create(   │
│   extra_body={        │
│     "program_id":"X"  │ ◄── 这是唯一一个新字段，paper Figure 8 称之为"全部改动"
│   } )                 │
└──────────┬────────────┘
           │ POST /v1/chat/completions
           ▼
 ┌────────────────────────────── ThunderAgent (FastAPI, app.py) ──────────────────────────────┐
 │                                                                                              │
 │  1. get_program_id(payload)         app.py:46-53        ← 从 body 或 extra_body 取 program_id│
 │  2. get_or_create_program(pid)      router.py:224-247   ← 若新则建 Program(state=ACTIVE,REASONING) │
 │  3. profile.on_request_arrive()     profile/state.py:106 ← 记录 tool_call_time 起点         │
 │  4. update_program_before_request   router.py:290-372   ← 容量检查/排队/migrate（核心）     │
 │     ├── default mode: 仅 least-loaded round-robin                                            │
 │     └── tr mode:                                                                             │
 │         • 若已 PAUSED → await waiting_event(超时 1800s 强复活)                               │
 │         • 若新程序而 queue 非空或容量不足 → 入 global_waiting_queue → 阻塞                   │
 │  5. profile.on_request_start()      profile/state.py:124 ← 记录 pause_time 结束              │
 │  6. proxy_request → vLLM/SGLang/SkyRL backend (vllm_request_processor.py)                   │
 │  7. on streaming: 每 20 tokens 回调 update_program_tokens_streaming （增量更新 c_p）         │
 │  8. on usage 到达: update_program_after_request → 把 status 从 REASONING 切回 ACTING        │
 │     同时用真实 prompt_tokens / context_len 校准 char_to_token_ratio（动量 0.2）              │
 │                                                                                              │
 │  并行运行的 _scheduler_loop (每 5 s):                                                        │
 │    a) backend.fetch_metrics()  ← Prometheus 文本解析 (vllm:* / sglang:*)                    │
 │    b) _greedy_resume()         ← BFD 装箱，从 global_waiting_queue 取回 paused 程序          │
 │    c) for each backend: if remaining_capacity()<0 → _pause_until_safe()                     │
 │                                                                                              │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 论文 → 代码对照精读

### 3.1 Program Abstraction（论文 §4.1，Table 1）

论文定义：`P = ⟨ID, c, 𝒯, ℒ, τ, s⟩`。仓库实现：

[ThunderAgent/program/state.py:33-47](ThunderAgent/program/state.py#L33)

```python
class ProgramStatus(Enum):       # 论文中的 τ ∈ {R, A}
    REASONING = "reasoning"      # τ = R, on-GPU prefill+decode
    ACTING    = "acting"         # τ = A, off-GPU tool / idle

class ProgramState(Enum):        # 论文中的 s ∈ {Active, Paused, Terminated}
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"

@dataclass
class Program:                   # 元组 P
    program_id: str              # ID
    backend_url: Optional[str]   # ℒ (current placement)
    origin_backend: Optional[str]# ← 用来支持 migration 的回退
    status: ProgramStatus        # τ
    state:  ProgramState         # s
    context_len: int             # 用 char/ratio 估计的 c (在还没收到 usage 前)
    total_tokens: int            # c (真实 KV-cache footprint)
    step_count: int              # paper 没显式建模，但用来区分 "new program (step=1)" vs 老程序
    profile: Optional[ProfileState]
    waiting_event: Optional[asyncio.Event]  # 实现 Pause/Restore 的 async coordination
    marked_for_pause: bool       # 见下文 — 实现"REASONING 不立即 pause"
    acting_since: Optional[float]# 记录进入 ACTING 的时刻，给 f(t)=2⁻ᵗ 用
```

注意论文里的 `𝒯`（tool environment set）**没有作为内部字段建模**—— 因为真正的 docker 生命周期由 agent 自己管，ThunderAgent 只通过 `/programs/release` 这个 hook 被通知。

### 3.2 Cost Model（论文 §4.2，公式 2-3）

论文：`Cost = ∫ M(t) dt = Cost_decode + Cost_prefill + Cost_recompute + Cost_unused + Cost_caching`。

代码里没有显式去算这五个积分项，而是把"瞬时 KV footprint"作为代理量，所有调度决策都用它：

[backend/state.py:92-143](ThunderAgent/backend/state.py#L92)

```python
@property
def reasoning_program_tokens(self):   # Σ c_p for τ=R (产生 Cost_decode/prefill)
    return sum(p.total_tokens for p in self._programs.values()
               if p.status == ProgramStatus.REASONING)

@property
def acting_program_tokens(self):      # Σ c_p for τ=A (产生 Cost_caching)
    return sum(p.total_tokens for p in self._programs.values()
               if p.status == ProgramStatus.ACTING)

@property
def active_program_tokens(self):
    # 论文公式(7) 的左边: Σ c_p|τ=R  + Σ c_q · f(t_q)|τ=A
    # tool_coefficient 在 CLI 里叫 --acting-token-weight, default=1.0
    return int(self.reasoning_program_tokens +
               self.tool_coefficient * self.acting_program_tokens)
```

`shared_tokens` 实现的是论文 KV-cache thrashing 公式里"prefix cache 节省"的修正项：

[backend/state.py:129-136](ThunderAgent/backend/state.py#L129) ＋ [backend/vllm_metrics.py:294-311](ThunderAgent/backend/vllm_metrics.py#L294)

```python
shared_tokens = max(0, reasoning_program_tokens
                       - kv_cache_usage_perc × total_capacity)
```

即"我以为每个 reasoning 程序的全量 token 都得占用，但实际 vLLM 报告的 KV cache 用量更小，差额就是 prefix-cache 复用省下来的"。这个值每次 scheduler loop 刷新一次，再被 `has_capacity / capacity_overflow / remaining_capacity` 用作扣减项。

### 3.3 Periodic Thrashing Detection（论文 §4.3.1 公式 6/7）

论文公式 (6)：`C_total < Σ c_p` （触发 pause）；公式 (7) 加上 `f(t)` decay。

代码 — 完全一对一：[scheduler/router.py:660-718](ThunderAgent/scheduler/router.py#L660)

```python
async def _scheduled_check(self):
    # 1. 拉一次 metrics（异步、非阻塞）
    for backend in self.backends.values():
        await backend.fetch_metrics()

    # 2. 先尝试 resume —— 给排队的优先级（含 acting-token decay）
    await self._greedy_resume()

    # 3. 公式 (6) thrashing 触发条件
    for url, backend in self.backends.items():
        if backend.cache_config and backend.remaining_capacity() < 0:
            await self._pause_until_safe(backend)
```

`remaining_capacity` = `total_capacity − (active_tokens − shared_tokens + buffer)`，其中 `buffer = BUFFER_PER_PROGRAM(=100) × #active programs`，对应论文里的"hysteresis window"。
论文把 λ_max 和 λ_min 都设成 1，代码也是 —— buffer 是唯一的 hysteresis。

### 3.4 Shortest-First Eviction（论文 §4.3.1 Definition 4.1，Lemma 4.1）

论文证明 `Cost_recompute ∝ c²`，所以最优策略是优先淘汰**最短**的程序（Appendix E.3 有 exchange-argument 证明）。

代码：[router.py:519-543, 685-714](ThunderAgent/scheduler/router.py#L685)

```python
def _get_acting_programs_sorted(self, backend_url, ascending=True):
    # ascending=True → 短的在前 (shortest-first)
    return sorted(programs, key=lambda x: x[1].total_tokens)

async def _pause_until_safe(self, backend):
    while backend.remaining_capacity() < 0:
        acting = self._get_acting_programs_sorted(backend.url, ascending=True)
        if acting:                              # 先淘汰 ACTING（off-GPU 立即生效）
            self._pause_program(*acting[0]); continue
        reasoning = self._get_reasoning_programs_sorted(backend.url, ascending=True)
        if reasoning:                           # 再 mark REASONING（不打断 GPU 推理）
            self._mark_program_for_pause(*reasoning[0]); continue
        break
```

`pause()` 对应论文公式 (5)：把 program 从 backend 摘除，加入 `global_waiting_queue`，置 `backend_url=None`、`state=PAUSED`、`origin_backend=旧位置`、新建 `waiting_event`：

[router.py:545-572](ThunderAgent/scheduler/router.py#L545)

注意：REASONING 程序**不立即 pause**，因为论文公式（5）要求 P 当前必须是 Active 状态才能 Pause，且 GPU 上的 in-flight prefill/decode 打断会浪费已做的工作 → 代码引入 `marked_for_pause` flag，把 token 数预先记到 `future_paused_tokens`（[router.py:574-586](ThunderAgent/scheduler/router.py#L574)）以便容量计算不会过度 over-pause，等下一次 `update_program_after_request` 把它切回 ACTING 时再真正 pause（[router.py:411](ThunderAgent/scheduler/router.py#L411)）。

### 3.5 Time-Decay Function f(t) = 2⁻ᵗ（论文 §4.3.1，Appendix E.1）

论文证明：在 memoryless tool-time 假设下，最优衰减是 `f(t)=e^{-λt}` (连续) 或 `f(t)=x⁻ᵏ` (离散)。仓库实现的是离散版 `f(t)=2⁻ᵗ`，**只用在 resume 决策上**（让长时间 acting 的程序"假装释放"出更多容量给排队的）：

[backend/state.py:196-214](ThunderAgent/backend/state.py#L196)

```python
def remaining_capacity_with_decay(self) -> int:
    now = time.time()
    acting_decayed = 0.0
    for p in self._programs.values():
        if p.status == ProgramStatus.ACTING and p.acting_since is not None:
            t = now - p.acting_since
            acting_decayed += p.total_tokens * (2.0 ** -t)   # f(t)=2^-t
    effective = int(self.reasoning_program_tokens + acting_decayed)
    return self.cache_config.total_tokens_capacity - (effective - self.shared_tokens + buffer)
```

注意 *pause* 路径用的是普通 `remaining_capacity()`（**不打折**），*resume* 路径用 `remaining_capacity_with_decay()`（**打折**） —— 这正是论文 §4.3.1 末尾"discount the effective weight of acting programs' tokens"的设计意图：避免空转 caching 占着容量。

CLI 开关：`--use-acting-token-decay`（[router.py:67-68](ThunderAgent/scheduler/router.py#L67)）。

### 3.6 Global Program-Aware Waiting Queue（论文 §4.3.2）

论文要点：所有 backend 共享一个 waiting queue，paused program 的 KV cache 反正要重算，所以 node-agnostic，可以放到容量最大的节点 → 跨节点 memory balance。

代码：[router.py:92-93, 719-844](ThunderAgent/scheduler/router.py#L92)

```python
self.global_waiting_queue: Dict[str, PausedInfo] = {}  # 全局共享
self.pause_resume_lock = asyncio.Lock()                # 选/弹 原子化
```

实际的 resume 用 **BFD (Best Fit Decreasing) bin-packing**——论文里没明写算法，但描述上等价：

[router.py:719-844](ThunderAgent/scheduler/router.py#L719) `_greedy_resume`：
1. 收集每个 backend 的 remaining capacity（开启 decay 时用 decay 版本）。
2. 按优先级 `REASONING(step>1) → NEW(step=1) → ACTING` 选满足 Σc_p ≤ Σcap 的最大子集。
3. **Largest-first** 排序 → 每次把最大的塞到 remaining 最多的 backend → 每塞一个就 re-sort backends。

这里有一个 paper 没写但代码做了的小优化：每个程序需要的不是 `c_p`，而是 `c_p + BUFFER_PER_PROGRAM (=100)`（decode headroom）。

### 3.7 Pause/Resume 的 async 协作

这是 paper 完全没讲、但是代码里最巧妙的工程细节。当一个请求被判定要 pause/排队时，请求协程**不能直接返回**给客户端（OpenAI client 在等响应），所以：

[router.py:565-572, 846-868](ThunderAgent/scheduler/router.py#L846)

```python
state.waiting_event = asyncio.Event()        # pause 时建
await asyncio.wait_for(state.waiting_event.wait(), timeout=1800.0)
# scheduler 在 _greedy_resume 里 state.waiting_event.set() ＋ 置 None
```

`update_program_before_request` 在协程内 `await` 这个 event；scheduler loop 异步触发 set，请求继续走到 backend。这是把"程序级 scheduler"绑到"HTTP 请求级 async"的关键 trick。

如果 30 分钟没人 resume（极端 backpressure），强制路由到 least-loaded backend 兜底（避免无限挂起）。

### 3.8 Tool Resource Management（论文 §4.4）

论文说："Hook-based garbage collector ... immediate teardown sequence, systematically reclaiming sandboxes, network sockets, and compute slots."

**代码里只实现了 hook 本身**，真正的 docker 销毁交给 agent：

ThunderAgent 侧 — [app.py:137-151](ThunderAgent/app.py#L137)：

```python
@app.post("/programs/release")
async def release_program(request: Request):
    program_id = payload.get("program_id")
    released = await router.release_program(program_id)   # 从 backends/queue/programs dict 全部清掉
```

`router.release_program` ([router.py:428-461](ThunderAgent/scheduler/router.py#L428)) 做的是 router 内部 bookkeeping：把 program 从 backend、global queue、`future_paused_tokens`、`programs` dict 全部移除。

真正的 docker 回收逻辑在 agent 侧。例如 mini-SWE-agent ([examples/inference/mini-swe-agent/src/minisweagent/run/extra/swebench.py:231-252, 365-415](examples/inference/mini-swe-agent/src/minisweagent/run/extra/swebench.py#L231))：

```python
def release_router_program(program_id, model):
    # 1. notify ThunderAgent
    requests.post(f"{router_admin_url}/programs/release",
                  json={"program_id": str(program_id)}, timeout=2.0)

# Inside process_instance() finally-block:
    if env and hasattr(env, "cleanup"):
        env.cleanup()                                          # docker container 关停
    if cleanup_images:
        subprocess.run([docker, "image", "rm", "-f", image_name])  # docker image 删除
        # ... 顺便清理 dangling layers
    release_router_program(str(instance_number), model)
```

OpenHands 在 [examples/inference/OpenHands/evaluation/benchmarks/swe_bench/run_infer.py:628](examples/inference/OpenHands/evaluation/benchmarks/swe_bench/run_infer.py#L628)、Harbor 在 [openhands.py:104-132](examples/datagen/harbor/src/harbor/agents/installed/openhands.py#L104)、ToolOrchestra 在 [eval_hle_local.py:683](examples/inference/ToolOrchestra/evaluation/eval_hle_local.py#L683) 都是同样的模式。

**没有实现的部分（与论文措辞之间的 gap）**：
- 没有任何 ThunderAgent 直接 docker SDK / kubectl 调用。
- README 提到的 `extra_body["docker_ids"]` 字段在仓库源码里**完全没用** — `grep -rn docker_ids ThunderAgent/` 是空的。看起来是 API 文档先行，实际工程里 docker 生命周期 100% 由 agent client 自己负责，ThunderAgent 只在 `/programs/release` 时把内部 bookkeeping 清掉。

### 3.9 Asynchronous Environment Preparation（论文 §4.4）

> "When a high-priority program (high S_restore) approaches the restore threshold, the system asynchronously restores its execution environment before the GPU memory is allocated."

**这一项在 ThunderAgent 代码里没有显式实现**。`_greedy_resume` 决定要 resume 谁之后是同步 `waiting_event.set()`，没有"提前 N 秒通知 agent 起 docker"的机制。论文里 Figure 6a 的 Env-prep 加速（4.8s → 0.3s）按现状只能依赖 agent 自己复用 docker（mini-SWE-agent 的做法）或 pre-pull。

### 3.10 多 backend 后端：metrics 抽象

论文要求"decouples scheduling from execution backends (e.g., vLLM/SGLang)"。代码里通过 `MetricsClient` 抽象类做这件事：

[backend/metrics_base.py](ThunderAgent/backend/metrics_base.py)

```python
class MetricsClient(ABC):
    @abstractmethod async def fetch_metrics(self) -> bool: ...
    @abstractmethod async def fetch_cache_config(self) -> bool: ...
    @abstractmethod def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int: ...
```

三个具体实现，**全部基于 HTTP/Prometheus 文本，不需要修改任何引擎源码**：

| Backend | 文件 | capacity 来源 | shared_tokens 计算 |
|---|---|---|---|
| vLLM | [vllm_metrics.py](ThunderAgent/backend/vllm_metrics.py) | `vllm:cache_config_info{block_size, num_gpu_blocks}` 标签 → `block_size × num_gpu_blocks` | `reasoning_tokens − kv_cache_usage_perc × capacity` |
| SGLang | [sglang_metrics.py](ThunderAgent/backend/sglang_metrics.py) | `/get_server_info` JSON 的 `max_total_num_tokens` | `reasoning_tokens − token_usage × capacity` |
| SkyRL | [skyrl_metrics.py](ThunderAgent/backend/skyrl_metrics.py) | 用每 engine 200k tokens 估算（无标准接口） | 类似，基于 `kv_cache_usage_pct` |

### 3.11 Profiling（论文 Figure 6a，端到端 latency 拆解）

[ThunderAgent/profile/state.py](ThunderAgent/profile/state.py) 把每个请求拆成 4 段时间，写 CSV：

| 字段 | 起 | 终 |
|---|---|---|
| `prefill_s` | request_start | first_token |
| `decode_s` | first_token | last_token |
| `pause_s` | request_arrive | request_start （= 在 waiting queue 等的时间） |
| `tool_call_s` | last_request_end | request_arrive （= 上一步 tool 执行 + idle） |

还顺便算 `kv_hit_rate = cached_tokens / prompt_tokens`（OpenAI v1 usage 里 `prompt_tokens_details.cached_tokens`，vLLM 已经支持）。

---

## 4. **是否侵入式修改 vLLM / SGLang？答：完全没有。**

我做了以下检查：

```
$ grep -rln "import vllm\|from vllm\|import sglang\|from sglang" ThunderAgent/
                                                                  ← 空
$ find ThunderAgent -name "*.patch" -o -name "*.diff"
                                                                  ← 空
$ find . -type d -name "vllm" -o -name "sglang" | grep -v examples
                                                                  ← 空
```

ThunderAgent 与引擎的耦合**纯粹通过两个 HTTP 接口**：
1. `POST {backend}/v1/chat/completions`（OpenAI 兼容） —— `vllm_request_processor.py` 转发请求。
   - 唯一改动：转发前 `remove_program_id` 把 `program_id` 从 body / extra_body 剥掉（[vllm_request_processor.py:82-94](ThunderAgent/scheduler/vllm_request_processor.py#L82)），因为 vLLM 不认。
   - streaming 时强制 `stream_options.include_usage=True` 才能拿到 token 计数。
2. `GET {backend}/metrics`（Prometheus 文本，vLLM/SGLang 都原生 export）。SGLang 额外用 `/get_server_info` 拿容量。

所以你**不需要 patch / 重新编译 / fork** vLLM 或 SGLang，开箱原版镜像就能跑 ThunderAgent。这也是论文 Appendix B 强调的"low-overhead adoption: only 3 changes required"。

`examples/rl_training/SkyRL/skyrl-train/skyrl_train/inference_engines/{vllm,sglang}/` 里有 `vllm_engine.py` 等文件，那是 **SkyRL 上游项目自己**的 vLLM 嵌入式调用代码（把 vLLM 当库用来做 RL rollout），与 ThunderAgent 的调度逻辑无关。

---

## 5. Sandbox 到底是怎么"实现"的？

论文里的 sandbox/tool lifecycle 在仓库里**不是一个模块**，而是三件事：

1. **Schema**：客户端在 `extra_body` 里塞 `program_id`（约定）。可选的 `docker_ids` 仅在 README 里建议（**未在源码里使用**）。
2. **HTTP hook**：`POST /programs/release {program_id}` —— ThunderAgent 收到就清自己内存里的 program。
3. **Agent 侧 cleanup 模板**：仓库提供了多个例子说明 agent 应该怎么写 finally-block。模式都是：
   ```python
   try:
       run_rollout(program_id)
   finally:
       env.cleanup()                                    # docker stop / kill
       subprocess.run(["docker", "image", "rm", "-f", img_name])
       requests.post(f"{router}/programs/release",
                     json={"program_id": program_id})
   ```

换句话说：**论文里的"hook-based garbage collector"在开源实现里是一个外部约定**，ThunderAgent 自己不接管 docker 守护进程，它只在请求结束时被通知清掉自己的 bookkeeping。所谓 4.2× disk memory savings（Fig 6a 绿色柱）来自于 agent 客户端**及时**调用了 `/programs/release` + 自己的 docker rmi —— ThunderAgent 的贡献是"提供这个钩子并保证 bookkeeping 一致"。

---

## 6. 文件清单（30 秒上手）

只想看核心，按这个顺序读：

| # | 文件 | 行数 | 看什么 |
|---|---|---|---|
| 1 | [`pyproject.toml`](pyproject.toml) | 21 | 依赖只有 `fastapi httpx uvicorn` |
| 2 | [`ThunderAgent/__main__.py`](ThunderAgent/__main__.py) | 83 | CLI args，整体能力一目了然 |
| 3 | [`ThunderAgent/app.py`](ThunderAgent/app.py) | 224 | 所有 HTTP 端点（chat / release / metrics / programs） |
| 4 | [`ThunderAgent/program/state.py`](ThunderAgent/program/state.py) | 48 | Program dataclass（≈论文 Table 1） |
| 5 | [`ThunderAgent/backend/state.py`](ThunderAgent/backend/state.py) | 279 | capacity 数学（has_capacity / overflow / decay） |
| 6 | [`ThunderAgent/scheduler/router.py`](ThunderAgent/scheduler/router.py) | 962 | 全部调度逻辑：pause/resume/BFD/scheduler_loop |
| 7 | [`ThunderAgent/scheduler/vllm_request_processor.py`](ThunderAgent/scheduler/vllm_request_processor.py) | 289 | SSE 解析 + usage 提取 |
| 8 | [`ThunderAgent/backend/vllm_metrics.py`](ThunderAgent/backend/vllm_metrics.py) | 344 | vLLM Prometheus 解析 |
| 9 | [`ThunderAgent/backend/sglang_metrics.py`](ThunderAgent/backend/sglang_metrics.py) | 332 | SGLang 解析 + server_info |
| 10 | [`ThunderAgent/profile/state.py`](ThunderAgent/profile/state.py) | 258 | 时序拆解（prefill/decode/pause/tool） |

全部业务代码加起来 ~2.8k 行。**没有 C++ / CUDA / patch / 内核代码**。

---

## 7. 跑起来的最小套件

```bash
# 1) 装一个 mock OpenAI 服务（例如另一个 fastapi 假装 vLLM）或真的拉 vLLM
# 2) 装 ThunderAgent
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# 3) 起 ThunderAgent，指向 vLLM
thunderagent --backend-type vllm \
             --backends http://localhost:8000 \
             --router tr \
             --port 9000 \
             --metrics --profile

# 4) 客户端把 base_url 切到 :9000，extra_body 加上 program_id 就行
```

如果只是想验证 router/scheduler 本身的正确性，可以不挂 vLLM，但 `fetch_cache_config` 会失败、`scheduling_enabled` 路径就会因为 `cache_config=None` 自动降级。下文 §8 会给一个**完全可在 CPU 上跑通**的 mock 验证。

---

## 8. 论文承诺 vs 开源实现 — 落差小结

| 论文承诺 | 仓库实现 | 备注 |
|---|---|---|
| Program abstraction | ✅ 完整 | `Program` dataclass + 状态机 |
| Periodic thrashing check (公式 6/7) | ✅ 完整 | `_scheduled_check` + decay |
| Shortest-First Eviction | ✅ 完整 | `ascending=True` 排序 |
| f(t) decay | ✅ 离散 2⁻ᵗ 版 | 论文支持连续 e⁻λᵗ，代码固定 2⁻ᵗ |
| Global waiting queue + cross-DP migration | ✅ 完整 | BFD 装箱 |
| OpenAI-compatible passthrough | ✅ 完整 | 唯一改动是 program_id |
| Multi-backend (vLLM/SGLang) | ✅ 完整 | MetricsClient 抽象 |
| Hook-based GC for sandboxes | ⚠️ Hook 实现，docker 操作在 agent 侧 | ThunderAgent 不直接管 docker |
| Asynchronous environment preparation | ❌ 未在 ThunderAgent 内实现 | 没有 pre-resume 通知机制 |
| Cost model 形式化 | ⚠️ 隐式 | 仅 STP 的瞬时项，未保留 ∫ M(t)dt 历史 |
| `extra_body["docker_ids"]` | ❌ README 提到但代码未用 | 见 §5 |
| 侵入式 vLLM/SGLang 修改 | ✅ 完全没有 | HTTP-only |

整体上**核心调度算法 100% 实现**，与论文一致；周边的 tool/sandbox 管理被设计成 *约定 + hook*，把脏活留给 agent 客户端——这正是 paper "simple" 的关键。
