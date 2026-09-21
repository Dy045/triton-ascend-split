# DynamicCVPipeline 主 Pass 专题：访存与计算分离 · 子专题 03 —— UBOverflowCheckerPass（UB 空间溢出检查与裁剪）

> 覆盖 `SeparateMemoryFromCompute` 的 **子 Pass 2：UBOverflowCheckerPass**（独立 Pass）
>
> 源码文件：[`UBOverflowChecker.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.cpp)、[`UBOverflowChecker.h`](../../../third_party/ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.h)
>
> 流水线位置：`SeparateMemoryFromCompute`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L84）内第二个子 Pass（主编排 L48）。在 `MarkGMLoadPass` 打完多缓冲标记之后，校验 UB 空间并裁剪超限标记。

---

## 一、这个子 Pass 在做什么

多缓冲是"空间换时间"：每加一层缓冲就多占一份 UB。`MarkGMLoadPass` 打的标记 + `AllocMultiCache` 前的缓冲可能**叠加超限**——UB 总共只有 248KB，若所有缓冲 × 深度 + 临时 tensor 的占用总和超出，编译器生成的地址会越界、硬件访存踩踏。

`UBOverflowCheckerPass` 的任务：**静态估算全部缓冲与 tensor 的 UB 占用；若超过 248KB，则按缓冲大小从大到小逐个"摘掉"多缓冲标记（降级为单缓冲），直到总占用回到安全线内**。

执行分四步：

1. **收集**（`collectBuffers` / `collectTensorEmpties`）：只统计 Vector scope 内带 `multi_buffer` 标记的 `memref.alloc`，以及所有静态 shape 的 `tensor.empty`；
2. **算尺寸**（`computeBufferSize` / `computeTensorSize` → `computeShapedSize`）：每块缓冲算三段尺寸——`originalSize`（原始字节数）→ `reducedSize`（`TileAndBindSubBlock` 把 dim0 分给两个子块、按 `ceil(dim0/2)` 折算）→ `alignedSize`（按 256-bit 对齐单元向上取整，模拟真实分配）；
3. **算占用**（`checkUBOverflow`）：`totalBits = Σ(缓冲 alignedSize × multiBufferCount) + Σ(tensor alignedSize)`；超限且存在可裁剪候选才进入裁剪；
4. **裁剪**（`pruneMultiBufferMarks`）：候选按 `alignedSize` **降序**，逐个删除 `MultiBufferAttr`，每删一个**全量重估**一次 `totalBits`，直到不再超限。**用户 hint 强制开的标记（`gm_load_hint`）不参与裁剪**。

**一句话**：这是多缓冲流水线的"资源安全阀"——保证重叠优化不会让 UB 溢出，超限时按"影响最小、收益最大"的顺序优雅降级。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `BufferInfo` | 缓冲信息（头文件 L43-L58）：`{kind(Annot/Unannot), allocOp, markOp, originalSize/reducedSize/alignedSize, multiBufferCount, forOp, fromHint}` |
| `TensorInfo` | tensor.empty 信息（L60-L65）：`{emptyOp, originalSize/reducedSize/alignedSize}` |
| `UBEstimateResult` | 估算结果 `{totalBits}` |
| `UBConstants::UB_SPACE_SIZE_BITS=2031616` | UB 总空间：248KB = 248×1024×8 bit（L74） |
| `K_SUB_BLOCK_DIM=2` | `TileAndBindSubBlock` 把 dim0 拆给 2 个子块，估算时 `reducedDim0 = ceil(dim0/2)`（L77） |
| `STRIDE_ALIGN_BYTES=32` | 跨步对齐 32 字节（L80，此处未直接用，属 `EnableStrideAlign` 的背景常量） |
| `ALIGN_UNIT_BITS=256` | 对齐单元 256-bit（32 字节），`alignedSize` 的向上取整单位（L83） |
| `getAlignUnit` | 每元素对齐单元数 = `256 / 元素位宽`（cpp L54-L59） |
| `collectBuffers` | 只收集 **Vector scope** 内 `memref.alloc`，从 `annotation.mark` 读 `multi_buffer` 深度；`fromHint` 由 `gm_load_hint` 判定并消费（L61-L98） |
| `collectTensorEmpties` | 收集 Vector scope 内静态 shape、非标量、无 0 维的 `tensor.empty`（L100-L132） |
| `computeShapedSize` | 三段尺寸计算（L134-L169），公式见 3.3 |
| `checkUBOverflow` | `totalBits = Σ alignedSize×multiBufferCount + Σ tensor.alignedSize`（L182-L196） |
| `collectPruneCandidates` | 候选：`kind==Annot && markOp && alignedSize>0 && !fromHint`（L198-L208） |
| `shouldPrune` | `totalBits > UB_SPACE` 且候选非空才剪（L210-L227） |
| `pruneMultiBufferMarks` | 降序删标记 + 全量重估（L229-L269），返回 `LogicalResult`（始终 success，非致命） |
| `findMarkOp` | 从 alloc 的 users 中找 `annotation.mark`（L46-L52） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（UBOverflowChecker.cpp L271-L285）

```cpp
void UBOverflowCheckerPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  if (failed(pruneMultiBufferMarks(module))) {
    LOG_DEBUG("pruneMultiBufferMarks failed (non-fatal)");  // 非致命
  }
}
```

**要点**：整个 Pass 把工作委托给 `pruneMultiBufferMarks`，且失败**不置 fallback**（注释明示 non-fatal）——裁剪只是"少优化"，UB 超限本身不产生错误 IR。

### 3.2 收集阶段：collectBuffers / collectTensorEmpties（L61-L132）

```cpp
SmallVector<BufferInfo> triton::collectBuffers(ModuleOp module) {
  module.walk([&](scope::ScopeOp scopeOp) {
    bool isCube = false, isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector))) return WalkResult::advance();
    if (!isVector) return WalkResult::advance();          // 只统计 Vector scope
    scopeOp.walk([&](memref::AllocOp allocOp) {
      BufferInfo buf;
      buf.allocOp = allocOp;
      buf.forOp = allocOp->getParentOfType<scf::ForOp>();  // 所属主循环（备用）
      annotation::MarkOp markOp = findMarkOp(allocOp);
      if (markOp) {
        if (auto attr = markOp->getAttrOfType<IntegerAttr>(hivm::MultiBufferAttr::name)) {
          buf.kind = BufferInfo::Kind::Annot;             // 有 multi_buffer 标记
          buf.markOp = markOp;
          buf.multiBufferCount = std::max<int64_t>(attr.getInt(), 1);
          buf.fromHint = markOp->hasAttr(CVPipeline::kGMLoadHintAttr);
          if (buf.fromHint) markOp->removeAttr(CVPipeline::kGMLoadHintAttr);  // 消费保护区标记
        }
      }
      buffers.push_back(buf);
    });
  });
  return buffers;
}
```

**要点**：
- **只统计 Vector scope**：UB 溢出风险主要来自 Vector 侧（CUBE 侧缓冲少且由 `ssbuffer` 管理），聚焦 Vector 简化估算；
- **无标记的 alloc 也收集**（`kind=Unannot`、`multiBufferCount=1`）：它们仍占用 UB，必须计入 `totalBits`；
- **`fromHint` 读取即消费**：`gm_load_hint` 属性读到后立刻移除——该属性只服务于本 Pass 的"保护区判定"，读完不留垃圾属性给下游；
- `collectTensorEmpties` 过滤条件：静态 shape、rank>0、无 0 维（L113-L117），否则无法静态估尺寸直接跳过。

### 3.3 尺寸估算：computeShapedSize（L134-L169）

```cpp
static void computeShapedSize(ShapedType shapedType, int64_t &originalSize,
                              int64_t &reducedSize, int64_t &alignedSize) {
  // 前置校验：静态 shape、非空、无 0 维、位宽合法
  int64_t numElements = 1;
  for (auto dim : shape) numElements *= dim;
  originalSize = numElements * bitWidth;                    // ① 原始位宽×元素数

  // ② TileAndBindSubBlock：dim0 分给两个子块，每个子块拿一半
  int64_t reducedDim0 = (shape[0] + K_SUB_BLOCK_DIM - 1) / K_SUB_BLOCK_DIM;  // ceil(dim0/2)
  int64_t reducedElements = numElements / shape[0] * reducedDim0;
  reducedSize = reducedElements * bitWidth;

  // ③ 对齐：最后一维按 256-bit 对齐单元向上取整
  int64_t lastDim = shape.size() == 1 ? reducedDim0 : shape.back();
  int64_t alignUnit = getAlignUnit(elementType);            // 256 / 位宽
  if (lastDim % alignUnit == 0) { alignedSize = reducedSize; return; }
  int64_t alignedLastDim = (lastDim + alignUnit - 1) / alignUnit * alignUnit;
  alignedSize = (reducedSize * alignedLastDim + lastDim - 1) / lastDim;   // 按新 lastDim 等比放大
  alignedSize = llvm::alignTo(alignedSize, ALIGN_UNIT_BITS);              // 整体 256-bit 对齐
}
```

**要点**：
- **三段式模拟真实内存布局**：`originalSize`（理论字节数）→ `reducedSize`（子块拆分后的实际承载量）→ `alignedSize`（对齐后的真实占用）；
- **dim0 减半**：`TileAndBindSubBlock` 把一个 block 的 dim0 拆给两个子块并行处理，每个子块只持一半，故按 `ceil(dim0/2)` 折算；
- **对齐逻辑**：最后一维长度若不是 `256/位宽` 的整数倍，按上界对齐并**等比放大**整块大小（`reducedSize × alignedLastDim / lastDim` 向上取整），最后整体再 `alignTo(256)`——两条对齐路径都要走；
- `getAlignUnit`：`256 / bitWidth`，如 fp16(16bit) → 16 个元素一单元、fp32(32bit) → 8 个元素一单元。

### 3.4 占用估算与裁剪决策：checkUBOverflow / collectPruneCandidates / shouldPrune（L182-L227）

```cpp
UBEstimateResult triton::checkUBOverflow(ArrayRef<BufferInfo> buffers, ArrayRef<TensorInfo> tensors) {
  for (const auto &buf : buffers)
    if (buf.alignedSize > 0) result.totalBits += buf.alignedSize * buf.multiBufferCount;  // 缓冲×深度
  for (const auto &tensor : tensors)
    if (tensor.alignedSize > 0) result.totalBits += tensor.alignedSize;                    // tensor 单份
  return result;
}

static SmallVector<size_t> collectPruneCandidates(ArrayRef<BufferInfo> buffers) {
  for (size_t index = 0; index < buffers.size(); ++index)
    if (buf.kind == Kind::Annot && buf.markOp && buf.alignedSize > 0 && !buf.fromHint)
      candidates.push_back(index);      // 只剪：自动打的标记、非 hint、可估尺寸
}

static bool shouldPrune(const UBEstimateResult &result, ArrayRef<size_t> candidates) {
  if (result.totalBits <= UBConstants::UB_SPACE_SIZE_BITS) return false;  // 安全
  if (candidates.empty()) return false;                                    // 无对象可剪
  return true;
}
```

**要点**：
- **多缓冲按深度累乘**：同一块缓冲 3 层就占 3 份对齐尺寸——这正是裁剪的"收益来源"；
- **hint 保护区**：`fromHint`（用户强制）的缓冲不进入候选，`collectBuffers` 已把 `gm_load_hint` 属性读走，此处只剩 `fromHint` 布尔可查；
- `alignedSize > 0` 排除无法静态估尺寸的缓冲（不可估就无法量化收益，不动它）。

### 3.5 裁剪主逻辑：pruneMultiBufferMarks（L229-L269）

```cpp
LogicalResult triton::pruneMultiBufferMarks(ModuleOp module) {
  // Step1: 一次收集，逐个算尺寸
  auto buffers = collectBuffers(module);
  for (auto &buf : buffers) computeBufferSize(buf);
  auto tensors = collectTensorEmpties(module);
  for (auto &tensor : tensors) computeTensorSize(tensor);

  // Step2: 初算占用 + 收集候选 + 决策
  auto result = checkUBOverflow(buffers, tensors);
  auto candidates = collectPruneCandidates(buffers);
  if (!shouldPrune(result, candidates)) return success();   // 安全/无对象直接结束

  // Step3: 按对齐尺寸降序（先摘大的，收益最大）
  llvm::sort(candidates, [&](size_t lhs, size_t rhs) {
    return buffers[lhs].alignedSize > buffers[rhs].alignedSize;
  });

  // Step4: 逐个摘标记，每摘一次全量重估
  int deleted = 0;
  for (size_t index : candidates) {
    auto &buf = buffers[index];
    buf.markOp->removeAttr(hivm::MultiBufferAttr::name);   // 摘掉 multi_buffer
    buf.kind = BufferInfo::Kind::Unannot;                  // 同步更新内存态
    buf.multiBufferCount = 1;                              // 降级为单缓冲
    ++deleted;
    result = checkUBOverflow(buffers, tensors);            // 全量重估
    if (result.totalBits <= UBConstants::UB_SPACE_SIZE_BITS) break;  // 安全即停
  }
  return success();
}
```

**要点**：
- **降序贪心**：`alignedSize` 最大者先摘——一次释放最多空间，用最少裁剪次数回到安全线，最大限度保留其余多缓冲重叠；
- **同步更新内存态**：删 mark 后立即改 `kind`/`multiBufferCount`，保证下一次 `checkUBOverflow` 用的是最新状态（所以是"全量重估"而非"局部减法"）；
- **不剪到 0 也安全返回**：即使所有候选摘完仍超限，也返回 success（宁可多缓冲全失效，不做错误裁剪/不报错）——降级行为由后端兜底。

---

## 四、流程总结

### 4.1 处理流水

```
UBOverflowCheckerPass（runOnOperation L271）
  └─► pruneMultiBufferMarks（L229）
        ├─► Step1 收集（只 Vector scope）
        │      ├─► collectBuffers：alloc + multi_buffer 标记 → BufferInfo（读走 gm_load_hint）
        │      └─► collectTensorEmpties：静态 tensor.empty → TensorInfo
        ├─► 尺寸：computeBufferSize/computeTensorSize
        │      └─► computeShapedSize：original → reduced(dim0 减半) → aligned(256-bit)
        ├─► Step2 估算与决策
        │      ├─► checkUBOverflow：totalBits = Σ(aligned×深度) + Σ(tensor)
        │      ├─► collectPruneCandidates：Annot && !fromHint && aligned>0
        │      └─► shouldPrune：超限 && 有候选
        ├─► Step3 按 alignedSize 降序排序
        └─► Step4 逐个摘 MultiBufferAttr → 更新 BufferInfo → 全量重估 → 安全即停
```

### 4.2 与上下游的协作

- **输入**：`MarkGMLoadPass` 打的 `annotation.mark + multi_buffer`（含 `gm_load_hint` 保护区）、Vector scope 内所有 alloc 与 tensor.empty；
- **产出**：被裁剪的 mark 移除 `multi_buffer`（缓冲降级为单缓冲）、`gm_load_hint` 属性被消费清空；
- **下游**：`RemoveSsbufAttr` 清理 `ssbuffer.*`；后端按剩余 `multi_buffer` 深度分配缓冲。

---

## 五、面试要点

1. **UBOverflowCheckerPass 的目标？** 静态估算 Vector 侧全部缓冲（×多缓冲深度）与临时 tensor 的 UB 占用；超过 248KB（2031616 bit）时按大小降序摘掉自动打的多缓冲标记，直到回到安全线，防止 UB 越界。

2. **为什么只统计 Vector scope？** UB 的主要压力在 Vector 侧（大量中间 buffer + 多缓冲），CUBE 侧缓冲少且由独立管理机制（ssbuffer）掌控；聚焦 Vector 即可覆盖主要风险，避免过度复杂。

3. **三段尺寸分别模拟什么？** `originalSize`=理论字节数；`reducedSize`=`TileAndBindSubBlock` 把 dim0 拆给两个子块、每块按 `ceil(dim0/2)` 折算后的承载量；`alignedSize`=按 256-bit 对齐单元对最后一维向上取整并等比放大、再整体 256-bit 对齐的真实占用。三段递增，贴近后端实际分配。

4. **对齐单元怎么算？** `getAlignUnit = 256 / 元素位宽`：fp16 → 16 元素、fp32 → 8 元素一单元。最后一维不足整单元时按上界对齐并等比放大整块，最后整体 `alignTo(256)`。

5. **totalBits 怎么算？** `Σ(缓冲 alignedSize × multiBufferCount) + Σ(tensor alignedSize)`。多缓冲按深度累乘是裁剪收益的来源；tensor 无多缓冲、只算一份。

6. **为什么按 alignedSize 降序裁剪？** 大缓冲一次摘掉释放的空间最多，用最少次数回到安全线，保留其余缓冲的重叠度——贪心策略下优化损失最小。

7. **为什么每摘一个就全量重估？** 摘掉一个标记会改变 totalBits，而多缓冲可能共享复用（同 for 内）；局部减法无法保证精确。基于更新后的 `BufferInfo` 列表整体重算，判断才可靠。

8. **为什么 hint 强制开的标记不剪？** 用户通过 `gm_load` 显式指定的多缓冲深度代表明确意图（性能优先），自动优化不能违背。`collectBuffers` 读走 `gm_load_hint` 属性后，`collectPruneCandidates` 用 `fromHint` 布尔排除它们。

9. **裁剪不完也返回 success 是为什么？** 摘完所有候选仍超限只说明该 kernel 本来放不下，此时保持单缓冲（全降级）比继续裁剪/报错更安全；本 Pass 定位是"尽力而为的资源平衡"，任何情况下都不应阻断后续 `RemoveSsbufAttr`。
