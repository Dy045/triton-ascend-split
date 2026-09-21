# DynamicCVPipeline 主 Pass 专题：访存与计算分离（01）

> 覆盖 `SeparateMemoryFromComputePass`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L84）
>
> 源码文件：
> - [`SeparateMemoryFromComputePass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SeparateMemoryFromComputePass.cpp)（编排主 Pass）
> - [`MarkGMLoadPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SeparateMemoryFromCompute/MarkGMLoadPass.cpp)（子 Pass 1：GM 加载标记）
> - [`UBOverflowChecker.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.cpp)（子 Pass 2：UB 空间溢出检查）
> - [`UBOverflowChecker.h`](../../../third_party/ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.h)（BufferInfo/TensorInfo/UBConstants 定义）
>
> 流水线位置：`AddControlFlowCondition`（L83）之后、`RemoveSsbufAttr`（L85）之前。此时多缓冲结构、同步条件、迭代次数均已就绪，本 Pass 做**访存路径的收尾**：给 GM→UB 加载打多缓冲标记（解耦访存与计算），并校验 UB 空间是否放得下所有缓冲。

---

## 一、这个专题在做什么

`AddControlFlowCondition` 已经把计算侧的多缓冲流水线建好了：每个 compute block 有了同步条件、主循环迭代次数已扩展。但还缺最后两块拼图：

1. **GM 加载路径没有多缓冲标记**。数据从全局内存（GM）拷进统一缓冲（UB）的 `memref.copy`，如果它也要跟计算重叠，就必须告诉后端"这块缓冲应该做几缓冲"。本模块第一个子 Pass `MarkGMLoadPass` 就是**识别这类拷贝并打上 `multi_buffer` 标记**（写在 `annotation.mark` 属性上）。
2. **UB 空间可能放不下**。多缓冲是"空间换时间"——每多一层缓冲就多占一份 UB。所有缓冲 + tensor 的 UB 占用可能超过 248KB。第二个子 Pass `UBOverflowCheckerPass` 做**静态估算**：超限时按缓冲大小从大到小**逐个摘掉多缓冲标记**（降级为单缓冲），直到总占用回到安全线内。

**两个子 Pass 的关系**：MarkGMLoad 负责"加标记"，UBOverflowChecker 负责"超限时减标记"。它们一起保证"访存与计算解耦"在**资源可行**的前提下发生。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `MarkCandidate` | GM 加载候选：`{copyOp, destAlloc, scopeOp, bufferCount, fromHint}`，由 `collectCandidate` 收集（MarkGMLoadPass.cpp L58-L64） |
| Rule 1 `traceSourceToFuncArg` | 源链穿透 view-like/extract_slice/for-iter_arg/while，必须终止于 **entry func 的 BlockArgument**（GM 指针），否则不是 GM 加载（L98-L156） |
| Rule 2 `traceDestToAlloc` | 目标链穿透 view-like/extract_slice 后必须是 `memref.alloc`（L161-L174） |
| Rule 3 `resolveBufferCount` | 按 scope 类型取默认多缓冲数：Vector=2（`kDefaultVBufferCount`）、CUBE=1（`kDefaultCBufferCount`）（L179-L196） |
| `resolveHintBufferCount` | 用户编译 hint：`gm_load` 属性值 0/1=强制关闭，>=2=强制开启+指定深度；多 hint 冲突返回 -1（L206-L235） |
| `kGMLoadMultiBufferHintAttr="gm_load"` / `kGMLoadHintAttr="gm_load_hint"` | 前者是用户 hint 属性（IntegerAttr），后者是标记来源标记（UnitAttr，Common/Utils.h L81-L82） |
| `markGMLoadCandidate` | 在 destAlloc 上插入/更新 `annotation.mark` + `hivm::MultiBufferAttr`，bufferCount<=1 则跳过（L257-L288） |
| `BufferInfo` | UB 缓冲信息：`{allocOp, markOp, kind(Annot/Unannot), originalSize/reducedSize/alignedSize, multiBufferCount, forOp, fromHint}` |
| `TensorInfo` | 未缓冲化的 `tensor.empty` 信息：`{emptyOp, originalSize/reducedSize/alignedSize}` |
| `UBConstants::UB_SPACE_SIZE_BITS=2031616` | UB 总空间 248KB 按 bit 计（2031616 = 248×1024×8），静态估算上限 |
| `K_SUB_BLOCK_DIM=2` | `TileAndBindSubBlock` 把 dim0 拆给两个子块，估算时 dim0 减半 |
| `ALIGN_UNIT_BITS=256` / `STRIDE_ALIGN_BYTES=32` | 对齐单位：256-bit 向上对齐 / 跨步 32 字节对齐 |
| `getAlignUnit` | 每元素可容纳的对齐单元数：`256 / 元素位宽`（UBOverflowChecker.cpp L54-L59） |
| `computeShapedSize` | 三段式尺寸：`originalSize = numElements×bitWidth` → `reducedSize`（dim0 减半后）→ `alignedSize`（256-bit 对齐）（L134-L169） |
| `checkUBOverflow` | `totalBits = Σ(缓冲 alignedSize×multiBufferCount) + Σ(tensor alignedSize)`（L182-L196） |
| `collectPruneCandidates` | 可裁剪候选：`kind==Annot` 且 **非 hint 来源** 且 `alignedSize>0`（L198-L208） |
| `shouldPrune` | 总占用 > UB_SPACE 且候选非空，才进入裁剪流程（L210-L227） |
| `pruneMultiBufferMarks` | 按 alignedSize 降序逐个删除 `MultiBufferAttr`，每次删除后**全量重估**，直到不超限（L229-L269） |

---

## 三、逐行讲解

### 3.1 主编排：runOnOperation（SeparateMemoryFromComputePass.cpp L37-L55）

```cpp
void SeparateMemoryFromComputePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;                         // 上游已 fallback，短路
  }

  OpPassManager pm(module.getOperationName());

  pm.addPass(createMarkGMLoadPass());        // 子 Pass 1：给 GM 加载打多缓冲标记
  pm.addPass(createUBOverflowCheckerPass()); // 子 Pass 2：UB 溢出检查与标记裁剪

  if (failed(runPipeline(pm, module))) {
    return;                                 // 失败仅返回，不置 fallback
  }
}
```

**要点**：
- 与 `AddControlFlowCondition`（失败置 `ERRCODE_FAILED`）不同，本 Pass **失败不写 fallback**（L50-L52），因为两个子 Pass 内部都已经把"无法处理"转换为自身的降级策略（跳过标记/裁剪），不阻断后续 `RemoveSsbufAttr`；
- 两个子 Pass 顺序依赖：必须先有 MarkGMLoad 打的 `multi_buffer` 标记，UBOverflowChecker 才能统计与裁剪。

### 3.2 与上下游的衔接

- **上游**：`AddControlFlowCondition` 产出的 if/同步/扩展循环结构，`alloc` 上的 `multi_buffer` 标记（`AllocMultiCache` 已打）、`gm_load` 用户 hint 属性；
- **下游**：`RemoveSsbufAttr`（L85）清理 `ssbuffer.*` 属性，随后后端编译器读取 `annotation.mark + multi_buffer` 决定实际缓冲深度。

---

## 四、流程总结

### 4.1 处理流水

```
SeparateMemoryFromComputePass
  ├─► fallback 短路（L40-L42）
  ├─► MarkGMLoadPass（子 Pass 1）
  │      ├─► Phase1 只读收集：walk memref.copy → Rule1(源=GM指针) ∧ Rule2(目标=alloc)
  │      │        → MarkCandidate（copyOp/destAlloc/scopeOp）
  │      ├─► Phase2 解析缓冲数：用户 hint（gm_load）优先；否则 Rule3 按 scope 取默认值
  │      │        （Vector=2 / CUBE=1），解析失败 → ERRCODE_FAILED
  │      └─► Phase3 打标记：destAlloc 上插/更新 annotation.mark + MultiBufferAttr(N)，
  │              hint 来源额外打 gm_load_hint（bufferCount<=1 跳过）
  └─► UBOverflowCheckerPass（子 Pass 2）
         ├─► collectBuffers（Vector scope 内带 multi_buffer 的 alloc）+ collectTensorEmpties
         ├─► computeBufferSize / computeTensorSize（original→reduced→aligned）
         ├─► checkUBOverflow 求 totalBits（缓冲×深度 + tensor）
         ├─► collectPruneCandidates（Annot 且非 hint）→ shouldPrune 判超限
         └─► pruneMultiBufferMarks：按 alignedSize 降序删 MultiBufferAttr，
               每删一个全量重估，直到 totalBits ≤ UB_SPACE
```

### 4.2 两个子 Pass 的协作关系

- MarkGMLoad 是"增量"，UBOverflowChecker 是"平衡"；
- **hint 标记是保护区**：`collectPruneCandidates` 明确排除 `fromHint` 的缓冲（L204），用户显式指定的多缓冲不会被自动裁剪；
- 裁剪只发生在"自动判定"的标记上（无 hint、按 scope 默认值打的）。

---

## 五、面试要点

1. **为什么需要本模块？** 计算侧流水线建好后，GM→UB 的加载仍是单缓冲的，访存与计算无法重叠；本模块先给 GM 加载打多缓冲标记（解耦的前提），再保证 UB 空间放得下（资源可行性）。

2. **两个子 Pass 怎么分工？** MarkGMLoadPass 识别"源是 GM 指针的 memref.copy"，在目标 alloc 上写 `multi_buffer=N`；UBOverflowCheckerPass 统计所有带标记缓冲与 tensor 的 UB 占用，超限时按大小降序**摘标记**降级。一加一减保证解耦可行。

3. **Rule 1/2/3 分别校验什么？** 源链必须终止于 entry func 参数（确认是 GM 指针而非核间 buffer）；目标链必须终止于 memref.alloc（确认是本地 UB 缓冲）；缓冲深度按 scope 类型取默认（Vector 天然要 2 缓冲隐藏访存延迟、CUBE 通常 1）。

4. **为什么源链要穿透 for/while 的 iter_arg？** 循环内 copy 的源往往是前一迭代的迭代变量（流水线轮换），穿透 iter_arg 后取其 **init 值**继续追溯，才能找到真正的 GM 指针；for 的 induction variable（arg 0）不能作为源。

5. **用户 hint 怎么生效？** `gm_load` 属性（IntegerAttr）：0/1 强制关闭（跳过标记），≥2 强制开启且指定缓冲深度；若同一 alloc 上出现多个冲突 hint 值则整体返回 -1（不信任）。hint 强制开的标记打上 `gm_load_hint`，UBOverflowChecker 裁剪时**跳过**它们（保护区）。

6. **UB 估算为什么分三段？** `originalSize`（原始字节数）→ `reducedSize`（`TileAndBindSubBlock` 把 dim0 拆成两个子块、每个子块分一半，故 dim0 按 `ceil(dim0/2)` 折算）→ `alignedSize`（每元素按 256-bit 对齐单元向上取整，模拟真实分配的对齐开销）。三段递增贴近后端真实内存布局。

7. **为什么裁剪按 alignedSize 降序？** 一块大缓冲（如 64KB×N 深）降级省下的空间远大于多块小缓冲，贪心"先摘大的"能以最少次数回到安全线，减少对流水线重叠度的破坏。

8. **为什么每次裁剪后要全量重估？** 摘掉一个标记会改变 `totalBits`，而多缓冲深度可能彼此影响（同一个 for 内共享缓冲复用），局部减法不精确；`checkUBOverflow` 基于最新 `BufferInfo` 列表整体重算，保证判断正确。

9. **主 Pass 失败为什么不写 fallback？** 两个子 Pass 的所有异常都已被降级策略吸收：MarkGMLoad 对不可识别候选直接跳过、UBOverflowChecker 超限不裁剪也无害（只是多缓冲不生效）。没有"错误"只有"少优化"，故仅返回 success/failure，不置 `ERRCODE_FAILED`。
