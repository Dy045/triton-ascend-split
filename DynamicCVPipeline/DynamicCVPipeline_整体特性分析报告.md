# DynamicCVPipeline 整体特性分析报告：设计与性能优化方法

> 基于对 `AddDynamicCVPipeline`（[AddDynamicCVPipeline.cpp](../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp)）下全部 10 个 Pass 的逐行分析（PreCheckAvailable / StandardizeOp / PlanComputeBlock / ComputeBlockOpt / SplitDataflow / AnalyzeDataFlow / AllocMultiCache / AddControlFlowCondition / SeparateMemoryFromCompute / RemoveSsbufAttr）整理而成。
>
> 本报告回答三个问题：**这条流水线到底优化什么？** 用什么**设计**实现？每一步**如何**贡献性能？

---

## 一、总览：这条流水线在做什么

**一句话目标**：把一个 Triton kernel 拆成 **CUBE（AIC）与 Vector（AIV）两个核**的异构软件流水线，通过**多缓冲轮换 + 条件同步 + 迭代扩展**，让矩阵计算（matmul）与向量计算、访存与计算**重叠执行**，以隐藏访存与核间通信延迟，换取端到端吞吐提升。

| 项目 | 内容 |
|---|---|
| 适用硬件 | 仅 91095（`AddDynamicCVPipeline.cpp` L109-L112） |
| 输入 | 编译 IR 中的 `scf.for/while` 主循环 + `scope::ScopeOp`（CUBE/Vector 双核作用域） |
| 产出 | 带多缓冲、同步条件、`annotation.mark` 的流水线化 IR |
| 本质 | 软件流水线（software pipelining）在异构双核上的工程实现 |

**核心性能来源**（三个"重叠"）：

1. **CUBE/Vector 双核重叠**：矩阵乘在 CUBE、其余在 Vector，两核并行执行，核间用 flag（flag_id）与 SSBuffer（共享内存计数）握手；
2. **多缓冲轮换重叠**：同一缓冲开 N 份，consumer 读第 k 份时 producer 写第 k+1 份，让"生产-消费"跨迭代错开，隐藏访存延迟；
3. **访存/计算分离重叠**：GM→UB 的加载打多缓冲标记，与计算解耦，把加载延迟藏进计算。

---

## 二、三段式架构：十个 Pass 的分工

十个 Pass 按职责可分**规划、拆分、收尾**三阶段，每阶段的产出成为下阶段的输入契约。

```
阶段一：规划（能不能做、怎么分块）
  PreCheckAvailable ─ 可行性预检，排除不适合的 kernel
  StandardizeOp ──── 把累加 matmul 拆成 matmul + add（消除归约共享，便于分块）
  PlanComputeBlock ─ 按 OpClassifier 把 op 分类成 CUBE/Vector，赋 block_id
  ComputeBlockOpt ── UB 合并/load 优化/块合并/if 分裂，让块形状利于流水

阶段二：拆分（数据流与缓冲）
  SplitDataflow ──── 打 block_id/flag_id、跨核传输与 sync、SeparateCVScope 拆双核作用域
  AnalyzeDataFlow ── 6 项可行性检查，不过 → fallback
  AllocMultiCache ── 核内/核间多缓冲分配，写 ssbuffer.cross_deps / intra_deps 属性

阶段三：控制与收尾（让流水真正转起来）
  AddControlFlowCondition ─ 6 子 Pass：克隆去共享 → 拆 iter_arg → 建 if → 建依赖 DAG
                            → 扩展循环 + PIPE_S → 构造真实同步条件 → 扩展迭代次数
  SeparateMemoryFromCompute ─ MarkGMLoad（GM 加载多缓冲）+ UBOverflowChecker（UB 防溢出）
  RemoveSsbufAttr ──────────── 清理全部 ssbuffer.* 中间属性，交付干净 IR
```

**设计要点——"先分析后变换"**：所有阶段都严格遵循"只读收集候选 → 改写"模式（如 `MarkGMLoadPass` Phase1 收集、Phase2/3 改写），且中间状态全部以**属性（attribute）**在 IR 上显式传递（`block_id`/`main_loop`/`cross_deps`/`intra_deps`/`multi_buffer` 等），每个 Pass 只读自己需要的属性、只写自己的产出——**Pass 间零全局状态，可独立注册、独立失败**。

---

## 三、性能优化的四个核心机制

### 3.1 双核异构划分（PlanComputeBlock + SplitDataflow）

- `PlanComputeBlock` 用 `OpClassifier` 把 op 按 core_type 分类：matmul/累加类 → CUBE，elementwise/reduce/搬运类 → Vector；
- `SplitDataflow` 的 `SeparateCVScope` 把同一个 kernel 的 CUBE 与 Vector op 拆进各自 `scope::ScopeOp`，并在块间插入 `sync_block_wait/set` 与 flag 传输；
- **收益**：两核并行，CUBE 长计算与 Vector 访存重叠，等价于"粗粒度指令级并行"。

### 3.2 多缓冲 + 条件同步（AllocMultiCache + AddControlFlowCondition）

这是整条流水线性能的核心，机制链完整：

1. **缓冲深度计算**（`UpdateLoopIterTimes::calculateFactor`）：
   - 核内依赖：`requiredBuffers = consumerIdx - producerIdx + 1`；
   - 跨核依赖：`runFirst` 时再 `-1`；`comsumerIdx < producerIdx` 的复杂拓扑保守不扩；
   - tensor 迭代依赖（先消费后生产）：方向反转 `producerIdx - consumerIdx + 1`；
   - 三路用**分数交叉相乘**（`a/x > b/y ⟺ a*y > b*x`）取最大，避免浮点精度。
2. **迭代次数扩展**：`new_ub = lb + step * (ceil(iter * requiredBuffers / x) + ifCount)`——缓冲数不够时，靠"时间换空间"把依赖距离在迭代轴上拉开；
3. **同步条件落地**（`UpdateConditionInfo`）：
   - 核内依赖：loop iter_arg 计数（输入 `>0` 才读、输出 `<producer_num` 才写）；
   - 核间依赖：**SSBuffer 共享内存计数**（Vector0 地址 0 起 / Vector1 1024 起，每槽 4B，volatile load/store 保证跨核可见）；
   - tensor 迭代依赖：`==1/==0` 判据；
   - **flowOpt 深度重叠**：缓冲充足时对 DAG 距离为 3 的 producer→consumer 对放宽条件（`counter >= upperBound OR counter >= lower + step*opt_num`），进一步压缩气泡。
4. **核间启动对齐**：主循环插入 `PIPE_S`（flag 15）wait/set，保证 CUBE/Vector 流水线同时起步。

**收益**：流水线重叠度由"依赖距离/缓冲数"精确控制，无同步则缓冲被踩踏，全同步则退化为串行——本设计用计数条件在两者间取得最优平衡。

### 3.3 访存/计算分离（SeparateMemoryFromCompute）

- `MarkGMLoadPass`：识别"源=GM 指针、目标=本地 alloc"的 `memref.copy`（穿透 view-like/iter_arg 追溯），按 scope 默认深度（Vector=2/CUBE=1）或用户 `gm_load` hint 打 `multi_buffer=N` 标记；
- **收益**：GM 加载路径也参与多缓冲，访存延迟被计算掩盖。

### 3.4 UB 资源约束（UBOverflowCheckerPass）

多缓冲是"空间换时间"，UB 只有 248KB（`UB_SPACE_SIZE_BITS=2031616`）。Pass 对 Vector scope 全部缓冲（×深度）+ tensor 做三段式估算（`originalSize → reducedSize(dim0 子块减半) → alignedSize(256-bit 对齐)`），超限则按 alignedSize 降序摘除自动标记（**hint 强制标记为保护区不摘**），每摘一次全量重估。

**收益**：性能优化被约束在硬件资源内，防止 UB 越界导致错误结果——这是"正确性优先"的工程取舍。

---

## 四、工程鲁棒性设计（性能优化的安全网）

性能优化必须"失败可用"，本流水线用三层机制保证：

| 机制 | 位置 | 作用 |
|---|---|---|
| Fallback 属性（errCode） | 每个 Pass 入口 `hasFallbackAttr` 短路 | 任一 Pass 失败即中止后续优化，IR 状态可恢复 |
| `ERRCODE_FAILED=1` / `IGNORED=2` / `TUPLE_PRELOAD_FAILED=3` | `Common/Utils.h` L96-L100 | 区分"硬错误"（必须回退）与"预期回退"（如 DAG 有环 = 不做流水也正确） |
| 重试机制 `MAX_RETRY_TIMES=2` | `AddDynamicCVPipeline.cpp` L124-L163 | 首次失败后 `FallbackHelper` 快照恢复，重跑时把核内缓冲数强制降为 2、核间降为 1（更保守），tuple preload 失败则关闭后重试 |
| `ScopedDiagnosticHandler` | L114-L122 | 吞掉流水线内部诊断噪音，避免误报错误 |

**设计哲学**：性能优化全部建立在"可回退"之上——优化失败只会退回标准编译，绝不会产出错误 IR。

---

## 五、整体流程一句话串讲

```
输入 IR
  → PreCheckAvailable（能优化吗？）
  → StandardizeOp（matmul 拆干净）
  → PlanComputeBlock + ComputeBlockOpt（怎么分块：CUBE/Vector、块形态）
  → SplitDataflow（拆双核、建传输与同步骨架）
  → AnalyzeDataFlow（可行性 6 查）
  → AllocMultiCache（分配多缓冲，写依赖属性）
  → AddControlFlowCondition（6 子 Pass：去共享 → 拆 arg → 建 if → 建 DAG → 扩循环 →
                              真实条件 → 扩迭代，双核流水正式成型）
  → SeparateMemoryFromCompute（GM 加载多缓冲 + UB 防溢出）
  → RemoveSsbufAttr（清理中间属性）
  → 交付给后端
  └─ 任一步失败 → setFallbackAttr → 主编排 restore 快照 → 可选重试 → 标准编译兜底
```

---

## 六、总结：设计亮点与性能方法论

1. **软件流水线是总纲**：所有 Pass 都围绕"生产-消费重叠"服务，多缓冲是空间、条件同步是控制、迭代扩展是时间；
2. **属性即契约**：Pass 间零全局状态，中间信息全部以属性显式传递，工程上可插拔、可测试、可单独回退；
3. **分层防御**：可行性预检（PreCheck/AnalyzeDataFlow）、运行时防护（条件同步/SSBuffer volatile）、资源约束（UBOverflow）、失败兜底（fallback/retry）四层保证"优化正确 > 优化激进"；
4. **参数化权衡**：缓冲深度、flowOpt 阈值（`CROSS_CORE_BUFFER_COUNT_THRESHOLD` 等）、重试缓冲数都是可调旋钮，让重叠度随 kernel 形态自适应。

**一句话总结**：DynamicCVPipeline 通过"双核异构划分 + 多缓冲轮换 + 条件同步 + 迭代扩展"把访存延迟藏进计算，用属性契约保证工程可维护，用 fallback/retry 保证优化失败也可正确编译——是一套"正确性优先、重叠度自适应"的 AI 编译优化流水线。
