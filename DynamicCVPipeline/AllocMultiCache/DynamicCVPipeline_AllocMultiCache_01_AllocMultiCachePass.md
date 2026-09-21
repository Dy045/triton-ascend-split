# DynamicCVPipeline 主 Pass 专题：多缓冲分配（01）

> 覆盖 `AllocMultiCachePass`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L82）
>
> 源码文件：
> - [`AllocMultiCache.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AllocMultiCache.cpp)（编排主 Pass）
> - [`AddMultiBufferInnerScope.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.cpp)（核内多缓冲）
> - [`AddMultiBufferOuterScope.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.cpp)（核间多缓冲）
>
> 流水线位置：`AnalyzeDataFlow`（L81）之后、`AddControlFlowCondition`（L83）之前。此时可行性审计已通过、数据流已按块拆分，本 Pass 开始**真正分配多缓冲并注入轮询控制流**，实现计算与访存的重叠。

---

## 一、这个专题在做什么

`SplitDataflow` 已经把数据流按 block 拆分好、`AnalyzeDataFlow` 也审计通过，但此刻跨块依赖还是**单缓冲**的：producer 块写入一块缓冲，consumer 块读完前 producer 不能重写，否则数据被覆盖。这意味着 producer 和 consumer 只能**串行**执行——完全违背 CV 流水线"计算与访存重叠"的目标。

`AllocMultiCache` 的职责就是**打破这个串行瓶颈**：对循环内跨块依赖分配 N 块缓冲（N 由 `BufferCountManager` 决定，通常是 2 即双缓冲），每次迭代轮换使用不同缓冲。producer 写空闲缓冲、consumer 读已写好的缓冲，两者就能流水线重叠执行。

**两类依赖、两个子 Pass 分而治之**（AllocMultiCache.cpp L54-L58）：

```cpp
OpPassManager pm(module.getOperationName());
// Step 1: 核内多缓冲（Inner Scope）
pm.addPass(createAddMultiBufferInnerScopePass());
// Step 2: 核间多缓冲（Outer Scope）
pm.addPass(createAddMultiBufferOuterScopePass());
```

| 子 Pass | 处理对象 | 分配位置 | 依赖来源 |
|---|---|---|---|
| `AddMultiBufferInnerScope` | Vector scope 内**主循环内部**的跨块依赖 | 循环前分配 N 块 UB 缓冲 | 块间 tensor 依赖（同一核内 CUBE↔Vector 数据流） |
| `AddMultiBufferOuterScope` | **循环外层** scope 的核间传输 | 传输组内分配双缓冲（TCB） | `SplitDataflow` 打的 `kTransferId` 传输组 |

**两个子 Pass 共用同一个模式**：

1. **收集**：识别出所有需要多缓冲的依赖/传输；
2. **分配**：在合适位置创建 N 块缓冲（替换原单缓冲的分配点）；
3. **轮询**：注入 `scf.if` 条件链，按迭代奇偶性（`iter % N == i`）选择当前迭代使用第 i 块缓冲；
4. **标记**：给新 op 打 `kIntraDeps` / `kCrossCoreDeps` / `kIntraBuffer` 等标记，供下游 Pass 识别。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `BufferCountManager` | 缓冲计数工具（`Common/BufferCountManager.h`）。`getBufferCountByType(DepType)` 返回缓冲数：`InterCore`（核间）= 双缓冲开关，`IntraCore`（核内）= 核内多缓冲块数 |
| outer scope / inner scope | **外层 scope** = 循环外顶层 scope（分配核间跨循环多缓冲）；**inner scope** = Vector scope 内部主循环（分配核内跨块多缓冲） |
| 跨块依赖 | producer 块产出的值被不同 block_id 的 consumer 块消费，需要多缓冲轮换避免覆盖 |
| `kTransferId`（`ssbuffer.transfer_id`） | `SplitDataflow` 给核间传输 op 打的标记，用于分组同一个传输的所有 op（wait/transfer/set + alloc/mark） |
| `kIntraDeps`（`ssbuffer.intra_deps`） | 核内依赖标记，数组 `[groupId, 0|1]`：`1`=producer（写缓冲的 copy），`0`=consumer（读缓冲的 to_tensor） |
| `kCrossCoreDeps` | 核间依赖标记，同样 `[tid, 0|1]` 语义，供下游 `AddControlFlowCondition` 识别 |
| `kIntraBuffer` | 标记核内多缓冲引入的 `scf.if`/`copy`/`to_tensor` op，供后续 Pass 识别缓冲读写结构 |
| `kBlockId`（`ssbuffer.block_id`） | 块 id，新注入的 op 必须继承所在块的 block_id，保证块归属一致 |
| `kIterCounter` | while 主循环注入的迭代计数 loop-carried 变量，用于多缓冲轮询奇偶性 |
| `kReservedPipeFlagId = 15` | flag id 15 预留给流水线同步（PIPE_S），不能分配给传输，可用范围 0-14 |
| `FlagIdManager` | flag 分配管理器（`Common/FlagIdManager.h`），`acquireId()` 取可用 flag，`MAX_FLAG_ID`=14，`INVALID_FLAG_ID` 表示耗尽 |
| 轮询条件 | `(iter / step) % N == i` → 选第 i 块缓冲。for 用归纳变量推导；while 用注入的迭代计数器 `% 2` |
| `getOutermostSsbufferId` | 多层嵌套 op 时找最外层 op 的 block_id，保证嵌套 op 都归属外层 block_id，避免同一块内错判跨块 |
| `TCB`（tightly_coupled_buffer） | 紧耦合缓冲，`hivm.tightly_coupled_buffer` 属性带 id，核间传输的物理缓冲 |

---

## 三、逐行讲解

### 3.1 主编排：AllocMultiCachePass（AllocMultiCache.cpp L44-L70）

```cpp
void AllocMultiCachePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;   // 上游已回退 → 直接短路

  OpPassManager pm(module.getOperationName());
  // Step 1: 核内多缓冲（Inner Scope）
  pm.addPass(createAddMultiBufferInnerScopePass());
  // Step 2: 核间多缓冲（Outer Scope）
  pm.addPass(createAddMultiBufferOuterScopePass());

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    if (!CVPipeline::hasFallbackAttr(module))
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

**要点**：
- **先内后外**：`Inner` 先做核内多缓冲（给 while 注入迭代计数器），`Outer` 后做核间多缓冲（复用/归一化 while 计数器）。若 `Inner` 已注入过 `kIterCounter`，`Outer` 的 `ensureWhileOpHasCounter` 会直接复用；
- **任何子 Pass 失败统一兜底**：置 `ERRCODE_FAILED` 回退标准编译；
- 若 `Inner` 已因 `i1Found`/`memrefFound` 置过 fallback，`Outer` 入口 `hasFallbackAttr` 直接 return。

---

## 四、流程总结

### 4.1 两个子 Pass 的流水

```
AllocMultiCachePass
  ├─► Step 1: AddMultiBufferInnerScope   （核内：Vector scope 主循环跨块依赖）
  │      │
  │      ├─► while 主循环先注入迭代计数器（bufNum>1 时）
  │      ├─► collectInnerBlockInfo     收集块 + 跨块依赖
  │      ├─► 检查 memref 依赖 / i1 tensor 依赖 → fallback
  │      ├─► Phase 1: clone empty+fill 到 consumer 块
  │      │         └─► rematerialize tensor 根标量依赖
  │      ├─► Phase 2: 重新收集依赖（clone 产生的新跨块引用）
  │      ├─► clone alloc_tensor 到 consumer 块
  │      ├─► insertBuffersBeforeLoop  循环前分配 N 块 UB 缓冲
  │      ├─► markScalarDeps           标量依赖打 dep_mark
  │      ├─► processTensorDependencies 每个跨块 tensor 依赖：
  │      │        producer 写分支 + consumer 读分支（if-else 链）
  │      └─► insertWhileCounterOps    while 计数器 +1（body 末尾）
  │
  └─► Step 2: AddMultiBufferOuterScope  （核间：循环外传输组双缓冲）
         │
         ├─► 预注入 while 迭代计数器（双缓冲模式时）
         ├─► Step 1: 收集传输分组（按 kTransferId 分组，推导方向/flag/chain）
         ├─► flag 预算检查（输入 flag>15 → IGNORED；输出超预算 → 降级单缓冲）
         ├─► 标记 llvm.load/store volatile 的 crossDeps
         ├─► Step 2: 创建输出缓冲（分配 TCB id，输出 flag 同步 op）
         └─► Step 3: 注入轮询控制流（scf.if 包 wait/transfer/set）
```

### 4.2 与前后 Pass 的协作

- **前置依赖**：`SplitDataflow` 打的 `kTransferId`、`kCrossCoreDeps`（Outer 用）；`PlanComputeBlock`/`ComputeBlockOpt` 打的 `kMainLoop`、`kBlockId`；`AnalyzeDataFlow` 通过审计（保证 flag 范围、无 memref 等）；
- **产出**：核内多缓冲结构（alloc 组 + producer/consumer if 链 + `kIntraDeps`/`kIntraBuffer`）、核间双缓冲结构（输出 alloc + 轮询 scf.if + 输出 flag + TCB + `kCrossCoreDeps`）；
- **后置**：`AddControlFlowCondition`（L83）消费 `kCrossCoreDeps` 生成真正的同步控制流；`SeparateMemoryFromCompute` 分离访存；`RemoveSsbufAttr` 清理所有 `ssbuffer.*` 属性。

---

## 五、面试要点

1. **AllocMultiCache 的目标是什么？** 打破跨块依赖的**单缓冲串行**瓶颈：分配 N 块缓冲（默认双缓冲），按迭代奇偶轮换，让 producer 写空闲缓冲、consumer 读已写好缓冲，实现计算与访存重叠。

2. **为什么分 Inner/Outer 两个子 Pass？** 两类依赖位置不同：**Inner** 处理 Vector scope 主循环**内部**的块间 tensor 依赖（同一核内数据流），在循环前分配 UB 缓冲；**Outer** 处理**循环外层**的核间传输组（`kTransferId` 分组），在传输组内分配 TCB 双缓冲。先内后外，避免内外缓冲分配互相干扰。

3. **缓冲数从哪里来？** `BufferCountManager.getBufferCountByType`：`InterCore` 决定核间是否双缓冲（>1 才启用）；`IntraCore` 决定核内多缓冲块数。这是可配置的硬件资源预算。

4. **轮询条件怎么构造？** for 主循环：`(iter / step) % N == i`（用归纳变量推导）；while 主循环：先注入 `kIterCounter` 循环承载计数变量，轮询条件为 `counter % 2 == 0`。`scf.if` 链按 `idx==0/1/...` 分支选择缓冲。

5. **Inner 为什么要两阶段收集依赖？** Phase 1 克隆 `empty+fill` 到 consumer 块后，克隆链的 `ins` 可能引用 producer 侧 tensor，产生**新的跨块引用**；scalar 重物化也会产生新链。因此 Phase 2 必须重新 `collectInnerBlockInfo` 才能让新引用进入多缓冲管线。

6. **while 的迭代计数器如何注入？** `setupWhileIterArgCounter` 重建 whileOp：old inits/result-types 追加 i32 counter（初值 0），保持原属性，替换旧 uses 后删除旧 while；随后 `insertWhileCounterOps` 在 body 末尾（counter 消费块的 block_id 位置）插入 `counter += 1` 并替换 yield 中的占位。

7. **Outer 的方向（C→V / V→C）怎么判定？** 看传输组的 sender transfer op 所在 scope：在 Vector scope → `V→C`；否则 `C→V`。无 sender 时用 receiver 的 scope 反推。方向决定 flag 复用键。

8. **输出 flag 如何分配/复用？** 同一 `(originalFlag, direction)` 的所有传输组**共享一个输出 flag**（同步位对齐）；`FlagIdManager.acquireId()` 分配，跳过与原 flag 相同的 id。flag id 15 预留流水线，输入 flag 超过 15 → `ERRCODE_IGNORED` 回退；输出超预算（>14）→ 降级单缓冲。

9. **Outer 为什么要把 wait/transfer/set 都包进 scf.if？** 双缓冲时每次迭代要么走原缓冲（奇偶=0），要么走输出缓冲（奇偶=1）。wait/transfer/set 三者**必须同步轮换**，否则"producer 换 flag 而 consumer 不换"会造成握手不匹配死锁。

10. **与 SplitDataflow 的分工？** `SplitDataflow` 负责**拆分数据流**（把块间交互拆成显式传输 + 同步 + flag），`AllocMultiCache` 负责**给拆分后的传输分配多缓冲实现轮换**。前者是"把依赖变成显式握手"，后者是"把握手升级为双缓冲流水"。
