# DynamicCVPipeline 子 Pass 专题：下沉与模式优化（03）

> 覆盖 `ComputeBlockOptPass` 阶段 C 的 5 个子 Pass：
> `SinkI1ProducersIntoUsers` / `FixpipeOpt` / `MoveLoadIntoUser` / `UnifyStoreBlock` / `ExpSubfPattern`
>
> 源码文件：
> - [`SinkI1ProducersIntoUsersPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SinkI1ProducersIntoUsersPass.cpp)
> - [`FixpipeOptPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/FixpipeOptPass.cpp)
> - [`MoveLoadIntoUserPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MoveLoadIntoUserPass.cpp)
> - [`UnifyStoreBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/UnifyStoreBlockPass.cpp)
> - [`ExpSubfPatternPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/ExpSubfPatternPass.cpp)
>
> 流水线位置：`ComputeBlockOptPass`（01 篇）中阶段 C（L66-L77）

---

## 一、这个专题在做什么

阶段 B（02 篇）解决的是"UB 依赖收紧 + 碎片块合并"。阶段 C 处理**数据流两侧的边界 op**——数据进入计算块时的 **load/声明** 侧，和数据离开计算块时的 **store** 侧，以及两类**硬件特化模式**：

| 子 Pass | 处理对象 | 核心动作 |
|---|---|---|
| `SinkI1ProducersIntoUsers` | i1（布尔）产生者 | 把 i1 产生 op **复制**到每个使用处（i1 极小，复制代价低） |
| `FixpipeOpt` | `matmul → cast/mul → slice → store` | 整条链并入 matmul 块 + **core_type 改为 CUBE**（启用 FIXPIPE 专用通道） |
| `MoveLoadIntoUser` | `alloc → copy → to_tensor` 加载链 | 把整条加载链（含依赖闭包）并入首个用户块 |
| `UnifyStoreBlock` | `store（materialize/hivm.store）` | 把 store + 视图链 + 标量依赖并入**生产它的 Vector 块** |
| `ExpSubfPattern` | `extf → subf → exp` | 把 exp（和可选 extf）并入 subf 所在块 |

**共同的本质**：本专题的 5 个 pass **全部是"block_id 改写"**，其中有 2 个 pass（`SinkI1ProducersIntoUsers`、`FixpipeOpt`）还会**物理移动或克隆 op**。区别于 02 篇：这里处理的都是**单个 op 链**（而非块级图），模式匹配更具体、动作更直接。

两个重要共享机制（在 02/03 多处出现）：
- `cloneScalarOpsForCrossBlockUses`（`FixpipeOpt`、`MoveLoadIntoUser`、`UnifyStoreBlock` 都用）：**跨块使用的标量 op 先克隆**，避免合并后依赖环。
- `updateBlockIdWithInner`：改写 block_id 时**递归覆盖内部 region 的嵌套 op**。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| i1（布尔张量） | `elemType.isInteger(1)` 的 tensor。常用于 mask。**数据极小**，跨块传输的固定开销（同步、缓冲）远大于数据本身 → 复制到每个使用处更划算 |
| FIXPIPE | CUBE 的固定流水线：`matmul 结果 → cast → 写 GM` 直接走 CUBE 侧专用硬件通道，不需要经过 Vector 核。**激活方式 = 把链上 op 的 `core_type` 改成 `"CUBE"`** |
| `bufferization.MaterializeInDestinationOp` / `hivm::StoreOp` | 两种"写回内存"的 op，`UnifyStoreBlock`/`FixpipeOpt` 都把它们视为 store 语义 |
| `to_tensor`（`bufferization::ToTensorOp`） | 内存 → 张量的桥。`MoveLoadIntoUser` 处理的加载链终点 |
| GM（Global Memory） | 全局内存。store 的目标必须是 GM（`collectViewOpsAndCheckGlobalMemory`）才值得优化 |
| `getFixpipePreQuantMode` | 判断 matmul 结果消费方式：走 cast（trunc）模式还是乘（quant 量化）模式 |
| `transSource` / `hasQuantScaleCompileHint` | `FixpipeOpt` 量化模式的前置：找到"从 tensor 到标量"的转换点，且量化的 scale 值带 `kInlinableQuantScaleAttr` 编译提示 |

---

## 三、逐行讲解

### 3.1 SinkI1ProducersIntoUsersPass（SinkI1ProducersIntoUsersPass.cpp）

**做什么**：把产生 i1 张量的 op（如 `arith.cmpi`、`arith.andi` 等）**下沉 + 复制**到每个使用它的计算块。源码描述（L63）："Sink i1-producing ops next to each of their i1 uses"。

**候选过滤（L72-L100）**：

```cpp
static bool isValidI1Producer(Operation *op) {        // L72-L88
  if (isa<scf::SCFDialect>(op->getDialect())) return false;   // 控制流 op 不可复制
  if (op->getNumResults() > 1) return false;                  // 单结果
  for (auto result : op->getResults())
    if (auto tensorType = dyn_cast<mlir::TensorType>(result.getType()))
      if (elemType.isInteger(1)) return true;                 // tensor 且元素是 i1
  return false;
}

static bool isPureAndRegionless(Operation *op) {      // L90-L100
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return false;  // 无嵌套 region
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!effects.empty()) return false;                        // 无内存副作用
  }
  return true;
}
```

**约束**：必须是**纯计算**（无副作用、无 region）且产生 i1 tensor 的单结果 op——复制它是安全的（重算结果相同）。

**核心逻辑（L123-L192）**——对每个 producer（逆序遍历 `llvm::reverse(producers)`，后收集的先处理）：

```cpp
// Step 1: 收集同一 block 内的消费者（getAncestorInBlock 归一化）
SetVector<Operation *> consumers;               // L124-L133
// Step 2: 消费者按拓扑序排列
auto orderedConsumuers = mlir::topologicalSort(consumers);    // L143

// Step 3: 第一个消费者——直接移动（或保持）
int consumerBlockId = bm.getBlockIdByOp(orderedConsumuers[0]);   // L145
if (consumerBlockId == -1) {
  // 第一个消费者是控制流节点：producer 保持不动（它本来就在自己块里）
  blockId2Producer.insert({bm.getBlockIdByOp(p), p});
} else if (bm.isSameBlock(p, orderedConsumuers[0])) {
  // producer 与第一个消费者已在同一块：无需动
  blockId2Producer.insert({consumerBlockId, p});
} else {
  p->moveBefore(orderedConsumuers[0]);            // 物理移到第一个消费者前
  bm.updateBlockId(p, consumerBlockId);           // 改归属
  blockId2Producer.insert({consumerBlockId, p});  // 登记"该块已有的 producer 实例"
}

// Step 4: 其余消费者——每块复制一份
for (auto consumerInblock : orderedConsumuers) {
  int consumerBlockId = bm.getBlockIdByOp(consumerInblock);
  if (consumerBlockId == -1) continue;            // 控制流消费者不处理
  if (!seenBlockIds.insert(consumerBlockId).second) {
    // 该块已有 producer 实例 → 只重定向 use
    info.value().replaceUsesWithIf(producer->getResult(id),
        [&](OpOperand &use) { return use.getOwner() == consumerInblock; });
    continue;
  }
  // 新块 → 克隆一份
  auto cloned = OpBuilder(consumerInblock).clone(*p);       // L182
  bm.updateBlockId(cloned, consumerBlockId);
  info.value().replaceUsesWithIf(cloned->getResult(id),
      [&](OpOperand &use) { return use.getOwner() == consumerInblock; });
}
```

**要点**：
- **逆序遍历 + 拓扑排序消费者**（L123、L143）：先处理"依赖最深"的 producer（避免先克隆的 producer 又被后面的克隆重复依赖）；消费者拓扑排序保证"移动到的位置"稳定。
- **第一个消费者特殊处理**（L145-L160）：producer 要么保持原位（消费者是控制流），要么**物理移动**（`moveBefore`）到第一个消费者前——只改 block_id 不够，i1 值必须真实出现在消费者块内。
- **其余消费者按块去重**（L169-L191）：同一个块有多个消费者时**只克隆一份**，块内其余消费者通过 `replaceUsesWithIf` 重定向到克隆结果（use 按 owner 精确匹配）。
- **`isSameBlock` 特判**（L151）：producer 已在第一消费者块内 → 不动，登记即可（后续同块消费者直接重定向到原 producer）。

**为什么 i1 值得复制**：i1 是单比特，复制一份几乎零成本；而跨块传递 i1 需要一个完整的跨块缓冲 + 同步。**"以计算换通信"**——用一次廉价重算省掉一次昂贵的跨块传输。

### 3.2 FixpipeOptPass（FixpipeOptPass.cpp）

**做什么**：识别 **FIXPIPE 可支持的硬件模式**，把整条链并入 matmul 的块并把 **core_type 改成 CUBE**——让 matmul 结果直接走 CUBE 的固定流水线写 GM，不经 Vector 核。源码描述（L81-L82）："Optimize matmul-cast-store pattern for fixpipe by setting core_type to CUBE"。

**两种模式（源码注释 L205-L215 / L283-L293）**：

```
Pattern 1（Cast 模式）：
  linalg.matmul → arith.truncf/f（f32→bf16、f32→f16、i32→i8）
      → tensor.extract_slice
      → bufferization.materialize_in_destination memref.subview(gm)

Pattern 2（Quant 量化模式）：
  linalg.matmul → arith.mulf/muli（乘一个标量 scale 做量化）
      → tensor.extract_slice（可选）
      → bufferization.materialize_in_destination memref.subview(gm)
```

**模式匹配 `matchFixpipePattern`（L405-L451）**：

```cpp
Value matmulResult = matmulOp.getResult(0);
if (!matmulResult.hasOneUse()) return false;                       // 单用户

// 穿透嵌套 scf.for 循环，找到第一个逃逸出循环的值（外层可观察值）
Value outerOutValue = getFirstResultAfterLoop(matmulResult, *matmulOp.getDpsInits().begin());  // L414-L415

auto outerOutOp = outerOutValue.getDefiningOp();
if (bm.getBlockIdByOp(outerOutOp) == -1) {                        // 若尚未分块
  if (llvm::failed(bm.markOpBlockId(outerOutOp))) return false;   // 尝试分配
}
targetBlockId = bm.getBlockIdByOp(outerOutOp);                    // 目标块 = matmul 所在块

if (!outerOutValue.hasOneUse()) return false;
if (isa<linalg::MatmulOp>(outerOutOp)) toMergeWithMatmul.insert(outerOutOp);  // matmul→matmul 链

auto matmulUser = *outerOutValue.getUsers().begin();
if (CVPipeline::getFixpipePreQuantMode(matmulUser).has_value()) {
  if (isFixpipeCastPattern(matmulUser, toMergeWithMatmul)) return true;   // 模式 1
} else if (isValidMul(matmulUser, matmulResult, toMergeWithMatmul)) {
  if (isFixpipeMulPattern(matmulUser, toMergeWithMatmul)) return true;    // 模式 2
}
return false;
```

**`getFirstResultAfterLoop`（L350-L392）**——本 pass 最精巧的辅助函数：
- 输入：`nowV`（matmul 结果）、`outerInValue`（matmul 的 DPS init）；
- 若 `outerInValue` 有定义者 → 说明 matmul 不在循环内（或循环已经穿透完），返回 `nowV`；(L352-L355)
- 否则 matmul 在一个 `scf.for` 内，且满足**循环携带值的单链更新**：blockArg 单用、用户恰好是定义 `nowV` 的 op、`nowV` 单用且被 yield 在对应 argIdx 位置（`getLoopCarriedArgIndex`）；(L361-L382)
- 递归向外：`outerInValue = forOp.getInitArgs()[argIdx]`，`nextSearchValue = forOp->getResult(argIdx)`。(L384-L385)

**设计意图**：matmul 通常位于嵌套循环里，其 `dps_init` 是循环携带的累加器。必须找到**第一个脱离循环、对外可见的值**，FIXPIPE 链才能真正作用在外层（循环展开后 matmul 结果最终写 GM 的地方）。

**模式 1 Cast：`isFixpipeCastPattern`（L216-L265）**：
- `truncOp` 结果**单用**；(L219-L222)
- 用户是 `tensor.extract_slice`，或直接是**另一个 matmul**（`matmul→trunc→matmul` 模式，trunc 是 DPS input → 直接收下 trunc）；(L225-L240)
- extract_slice 结果单用 → `materialize_in_destination`；(L242-L256)
- **`isStoreToGM`（L184-L203）**：store 目标是 GM 的 subview（`collectViewOpsAndCheckGlobalMemory` 验证）。

**模式 2 Quant：`isFixpipeMulPattern`（L294-L321）**：
- mul 结果单用（`getOneUserExceptMarkOp` 忽略 `annotation.MarkOp`——量化提示标记不算真用户）；(L297-L301)
- 用户是 extract_slice（可选）→ store；(L302-L314)
- `isValidMul`（L144-L182）校验乘数：要么是标量（`transSource` 沿定义链收集到 tensor→scalar 转换点），要么是 `linalg.fill` 的常量填充；**且 op 必须有 `kInlinableQuantScaleAttr` 编译提示**（`hasQuantScaleCompileHint` L132-L142）——量化的 scale 必须可内联。

**`transSource`（L105-L130）**：从标量值沿定义链回溯，遇到 `tensor.extract`、已在集合内、跨 block 时停止——收集"tensor→标量"转换链上的所有 op，它们要随 matmul 一起标 CUBE。

**应用 `applyFixpipeOpt`（L453-L493）**：

```cpp
// 环检测（块级）
if (CVPipeline::willCreateCycle(matchedOps, memGraph, targetBlockId, bm)) return false;

for (Operation *op : matchedOps) {
  if (isa<scf::SCFDialect>(op->getDialect())) {
    op->walk([&](Operation *nestedOp) {
      if (CVPipeline::isSyncOp(nestedOp)) return WalkResult::advance();  // 同步 op 保留独立块！
      bm.updateBlockId(nestedOp, targetBlockId);
      nestedOp->setAttr(kCoreType, StringAttr::get(ctx, "CUBE"));
      return WalkResult::advance();
    });
  } else {
    bm.updateBlockId(op, targetBlockId);
    op->setAttr(kCoreType, StringAttr::get(ctx, "CUBE"));
  }
}
return true;
```

**关键正确性约束（L476-L479）**：**同步 op（sync）绝不并入计算块**——它必须保留自己的独立 block_id，否则前后 fence 会在 SplitDataflow 里被吞掉。

**入口 `runOnOperation`（L499-L551）**：
1. walk 所有 matmul，匹配模式，收集 `allMatchedPatterns`；(L515-L521)
2. **防环预处理（L531-L548）**：对每个模式，先用 `SplitIf::ScalarClosure` 收集标量闭包并入 `matchedOps`，再 `cloneScalarOpsForCrossBlockUses`——标量 op 被模式外使用会导致合并成环（`A→B→C、A↘D↗` 结构），克隆 C' 给 D 破环；
3. 逐个 `applyFixpipeOpt`；环检测失败则跳过该模式（LOG_DEBUG 记录）。

### 3.3 MoveLoadIntoUserPass（MoveLoadIntoUserPass.cpp）

**做什么**：把 `alloc → memref.copy → to_tensor` 的**加载链**（从 GM 读数据的完整链路）整体移入**第一个消费者块**，消除"数据已就位、消费在别块"的割裂。

**模式匹配 `matchLoadPattern`（L179-L220）**：

```cpp
// Step 1: copy 的 source 必须来自 GM
SetVector<Operation *> sourceViewOps;
if (!CVPipeline::collectViewOpsAndCheckGlobalMemory(info.copyOp.getSource(), sourceViewOps))
  return false;

// Step 2: copy 的 target 必须是 alloc（或从 alloc 来的视图）
Value dest = info.copyOp.getTarget();
if (auto allocOp = getOriginAlloc(dest)) {        // getOriginAlloc 沿视图链回溯到 alloc
  info.allocOp = allocOp;
} else return false;

// Step 3: 找 alloc 的 to_tensor 用户
for (auto *user : info.allocOp->getUsers()) {
  auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
  if (toTensor) { info.toTensorOp = toTensor; break; }
}
if (!info.toTensorOp) return false;

// matchedOps = {alloc, copy, toTensor}             // L215-L217
return true;
```

`getOriginAlloc`（L166-L177）：target 本身是 alloc，或沿 `ViewLikeOpInterface` 的 `getViewSource()` 链回溯到 alloc——处理 `alloc → subview → copy` 的间接形态。

**找首个消费者 `getFirstUser`（L95-L140）**：

```cpp
for (auto *user : toTensorOp->getUsers()) {
  auto *userInBlock = CVPipeline::getAncestorInBlock(user, toTensorOp->getBlock());
  if (userInBlock) {
    // 穿透同 block_id 的 collapse 链（collapse 是纯 reshape，必须跟着走）
    auto calUser = traceCalculateUser(userInBlock, bm, bm.getBlockIdByOp(toTensorOp), userChain);
    if (!firstUser || calUser->isBeforeInBlock(firstUser))
      firstUser = calUser, fistUserChain = userChain;   // 取 IR 顺序最早的
  }
}
// 约束：firstUser 必须 VECTOR 且 block_id ≠ to_tensor 的块
if (getOpCoreType(firstUser) != VECTOR_ONLY) return nullopt;   // L123
if (blockId == bm.getBlockIdByOp(toTensorOp)) return nullopt;  // L134 已在同块
matchedOps.append(fistUserChain);                              // collapse 链并入
```

- **`traceCalculateUser`（L75-L93）**：沿 `tensor.collapse_shape` 单链穿透（要求每环单用），直到遇到非 collapse 或跨块的 op——collapse 是纯 reshape，**必须随加载链一起搬**。
- 首个消费者必须是 VECTOR 且在不同块（否则无需移动）。

**依赖闭包 + 应用（runOnOperation L244-L310）**：

```cpp
// 收集同块依赖闭包（递归，块内且同 commonBlockId 的源）
SetVector<Operation *> opsToMove;
for (auto *op : matchedOps)
  collectAllDependencies(op, opsToMove, commonBlockId, bm, memGraph);   // L290-L293

// 防环：跨块使用的标量先克隆
CVPipeline::cloneScalarOpsForCrossBlockUses(bm, opsToMove, bm.getBlockIdByOp(firstUser));

// 环检测 + 整体迁移
if (CVPipeline::willCreateCycle(opsToMove, memGraph, targetBlockId, bm)) continue;
for (auto *op : opsToMove)
  bm.updateBlockIdWithInner(op, targetBlockId);    // 带内部 region 递归
```

`collectAllDependencies`（L222-L242）：从每个 matched op 出发，用 `DependencyHelper` 递归收集**同 block_id 且 = commonBlockId** 的源——加载链的依赖必须一起走，否则搬走 alloc 后它的源还在原块。

**与 02 篇 `UnifyAllocBlock` 的区别**：`UnifyAllocBlock` 把 alloc 并入"copy 所在块"（消费侧）；`MoveLoadIntoUser` 把"alloc+copy+to_tensor 整条链"并入"to_tensor 的首个消费者块"（计算侧）。二者方向相反，处理不同错位。

### 3.4 UnifyStoreBlockPass（UnifyStoreBlockPass.cpp）

**做什么**：把 **store 语义 op**（`materialize_in_destination` / `hivm.store`）并入**生产它的 Vector 计算块**。store 是"数据离开计算块"的出口，让它紧贴生产者，避免"计算完 → 跨块 → 再写"的多余往返。源码描述（L307-L309）："Merge store-semantic operations into the producer vector compute block"。

**`traceProducerOp`（L96-L122）**——从 store 的 source 回溯找生产者：

```cpp
Value cur = getStoreSource(storeOp);
while (Operation *defOp = cur.getDefiningOp()) {
  if (isViewOrExtractSliceOp(defOp)) {          // 视图透明，穿透
    dataViewOps.push_back(defOp);
    cur = getViewSourceValue(defOp);
    continue;
  }
  if (CVPipeline::isScalarLike(cur)) return nullptr;   // 标量链 → 不可合并
  if (getOpCoreType(defOp) != getOpCoreType(storeOp)) return nullptr;  // core 类型必须一致（VECTOR）
  return defOp;                                 // 命中真实计算 op
}
return nullptr;   // 到 block 参数（iter_arg / GM 参数）→ 不可合并
```

**约束**：store 的生产者必须是**同 core 类型（VECTOR）的真实计算 op**；纯视图可穿透；标量链/block 参数没有"生产者块"可并。

**`matchStorePattern`（L236-L269）**：
- store 必须有 block_id；(L240-L244)
- 生产者必须有 block_id **且与 store 同 block**（L255-L260）；
- `matchedOps = {producer} + {store} + viewOps + scalarDeps`——**producer 必须是第一个**（L262-L267，`cloneScalarOpsForCrossBlockUses` 靠它识别目标块）。

**收集统一对象**：
- `collectViewOpsToUnify`（L130-L153）：数据链的 view op + store dest 链上的 view op（同 block 才收）；
- `collectScalarDeps`（L165-L200）：递归收集**同 block 的标量依赖**（含 store 的控制流祖先 op——`scf.for` 的 lb/ub/step 等）。**外层块标量跳过**（L190-L192）：func 级常量被多块共享，搬动不安全。

**`applyStoreUnify`（L276-L294）**：以 producer 的块为目标，`willCreateCycle` 通过后整组 `updateBlockId`。

**入口 runOnOperation（L312-L359）**：
- Phase 1（L323-L328）：只收集 **VECTOR-core** 的 store op；
- Phase 2（L335-L355）：每个模式先 `cloneScalarOpsForCrossBlockUses`（破环），再 `applyStoreUnify`。

**与 `FixpipeOpt` 的关系**：`FixpipeOpt` 在流水线**之前**执行（L69 vs L73）。`FixpipeOpt` 已把 fixpipe 链标成 CUBE，因此这些 store 不再是 VECTOR（进不了 `UnifyStoreBlock` 的候选）；剩下**普通 Vector store** 才由 `UnifyStoreBlock` 处理。分工清晰：FIXPIPE 走 CUBE，普通 store 并入 Vector 生产者。

### 3.5 ExpSubfPatternPass（ExpSubfPatternPass.cpp）

**做什么**：匹配 **`extf → subf → exp`** 模式，把 `exp`（和可选的 `extf`）的 block_id 统一到 `subf` 所在块。`exp` 是 attention softmax 的核心计算，让它与减法同块可避免跨块传 exp 输入。

**`processSubfOp`（L112-L145）**：

```cpp
int subfBlockId = bm.getBlockIdByOp(subfOp);

// 1) subf → exp：subf 结果单用且用户是 exp，同 block
auto expOp = matchExpFromSubf(subfOp);          // L43-L61
if (!expOp) return;
int expBlockId = bm.getBlockIdByOp(expOp);
if (expBlockId != subfBlockId)
  bm.updateBlockId(expOp, subfBlockId);          // 把 exp 并到 subf 的块

// 2) extf → subf：subf 的两个操作数都来自 extf（可选）
auto extfOps = matchExtfFromSubf(subfOp);       // L88-L110
if (!extfOps) return;
for (auto extfOp : *extfOps)
  if (extfBlockId != subfBlockId)
    bm.updateBlockId(extfOp, subfBlockId);
```

**匹配细节**：
- `matchExpFromSubf`（L43-L61）：subf 结果**单用**、用户是 `math.exp`、**同 block**；
- `matchExtfFromSubf`（L88-L110）：subf 的 lhs/rhs 都来自 `arith.extf`；
- `isValidExt`（L63-L86）：extf 与 subf 同 block；输入 `f16`、输出 `f32`（fp16→fp32 扩宽）；extf 结果单用且用户恰是 subf。

**设计意图**：`extf(subf(extf(x), extf(y))) → exp` 是 fp16 输入下 softmax 的典型形态。subf 的 block_id 是"数据主链"的归属，把 exp/extf 拉进来让**整条 softmax 链同块**，减少跨块边界。整个 pass 无环检测——因为 exp/extf 都是 subf 的**直接下游/上游**，方向单一，天然无环（注意：这里只做 `updateBlockId`，且 exp/extf 是 subf 的邻近节点，成环风险低）。

---

## 四、算法流程总结

```
03_SinkAndPatternOpt（5 个子 Pass）
├─ 3.1 SinkI1ProducersIntoUsers   纯计算 i1 产生者 → 复制到每个用户块
│    找i1生产者(纯/无region/单结果) → 逆序收集消费者(拓扑序)
│    → 首消费者: 物理moveBefore+改id / 其余: 按块克隆+replaceUsesWithIf
├─ 3.2 FixpipeOpt                 matmul→cast/mul→slice→store(GM) → 并进matmul块+CUBE
│    穿透循环(getFirstResultAfterLoop) → cast模式/quant模式
│    → 标量闭包+跨块克隆破环 → 环检 → updateBlockId+core_type=CUBE(同步op除外)
├─ 3.3 MoveLoadIntoUser           alloc→copy→to_tensor 加载链 → 移入首个VECTOR用户块
│    GM源校验 → 找alloc → 找to_tensor → 首个用户(穿透collapse链)
│    → 依赖闭包 → 跨块标量克隆 → 环检 → updateBlockIdWithInner
├─ 3.4 UnifyStoreBlock            VECTOR store → 并入其生产者计算块
│    回溯生产者(穿透视图, core必须一致) → {producer,store,视图,标量依赖}
│    → 跨块标量克隆 → 环检 → updateBlockId
└─ 3.5 ExpSubfPattern             extf→subf→exp → 全部并入subf块
      exp单用同块 + extf(f16→f32)校验 → updateBlockId(无环检)
```

**整体策略**：阶段 C 的 5 个 pass 分两类——**复制型**（`SinkI1ProducersIntoUsers`，以复制换通信）和**搬迁型**（其余 4 个，把单 op 链并入主块）。共同点是：每个模式都**单链、简单、方向明确**，靠 `willCreateCycle` + `cloneScalarOpsForCrossBlockUses` 守住正确性。相比阶段 B 的块级图优化，阶段 C 是**局部的、模式的**优化。

---

## 五、面试要点

1. **SinkI1ProducersIntoUsers 为什么对 i1 用"复制"而不是"移动"？**
   i1 是单比特数据，复制一份（重算一次比较/与运算）几乎零成本；而跨块传 i1 需要一个跨块缓冲 + 同步，固定开销远大于数据本身。**以廉价计算换昂贵通信**。只有**纯、无副作用、无 region、单结果**的 op 才允许复制——复制要保证可重算。

2. **SinkI1ProducersIntoUsers 里"第一个消费者移动、其余消费者克隆"的设计？**
   第一个消费者处用 `moveBefore` **物理移动**（producer 必须真实出现在消费块内）；其余消费者按 **block_id 去重**——同块多个消费者共享一份克隆（`replaceUsesWithIf` 精确重定向），不同块各克隆一份。避免重复克隆 + 保证每块都有实例。

3. **FixpipeOpt 的本质操作是什么？为什么"改 core_type"就能启用硬件通道？**
   本质 = 把 `matmul→cast/mul→slice→store` 链上所有 op 的 block_id 并入 matmul 块，且 `core_type` 改为 `"CUBE"`。因为 FIXPIPE 是 CUBE 侧硬件，整条链标记 CUBE 后，下游 lowering 会为它生成 fixpipe 指令。这体现了 CV 流水线**"core_type 是可重标的软决策"**——按硬件能力修正 Plan 阶段的分配。

4. **FixpipeOpt 里"同步 op 绝不并入计算块"的约束为什么重要？**
   `applyFixpipeOpt` 对嵌套 region 的 op 递归改写 block_id 时，遇到 sync op 会 `WalkResult::advance()` 跳过。因为同步 op 承担"块间 fence"职责，必须保留**唯一独立 block_id**，否则 SplitDataflow 降栅栏时 fence 丢失，块间数据竞争。

5. **`getFirstResultAfterLoop` 在解决什么问题？**
   matmul 通常嵌套在 `scf.for` 里，结果被循环携带更新。FIXPIPE 链作用于**外层可观察值**。该函数沿"循环携带单链更新"（blockArg 单用 → 单 op 更新 → 单 yield 回传）逐层外推，找到第一个脱离循环的结果——**穿透循环拿"最终写 GM 的那个值"**，而不是循环体内中间值。

6. **MoveLoadIntoUser 与 UnifyAllocBlock（02 篇）的区别？**
   `UnifyAllocBlock` 把 `alloc/if(fill)/subview` 并入 **copy 所在块**（消费侧），目标是"声明跟着首次使用走"；`MoveLoadIntoUser` 把 `alloc→copy→to_tensor` **整条加载链 + 依赖闭包**并入 **to_tensor 的首个用户块**（计算侧），目标是"数据从 GM 读出后直接进入计算块"。一个是"声明归位"，一个是"加载链归位"。

7. **为什么 MoveLoadIntoUser/UnifyStoreBlock/FixpipeOpt 都要先 `cloneScalarOpsForCrossBlockUses`？**
   待搬 op 可能依赖**跨块共享的标量**（如常量、循环参数）。若直接搬走，标量被外部块继续使用会造成跨块依赖环（`A→B→C` 加 `A↘D↗` 结构）。先给外部使用方克隆一份标量，打破共享，再搬主链。**以复制破环**——与 MergeComputeBlock 的跨 Cube 克隆同思想。

8. **UnifyStoreBlock 对"生产者"有什么约束？为什么？**
   ① 生产者必须是**同 core 类型**（VECTOR）——store 并入 CUBE 块会改变 core 语义；② 生产者必须有 block_id 且与 store 同 block；③ 纯视图可穿透，但标量链/block 参数（iter_arg、GM 参数）无生产者可并，放弃。核心是"store 必须能并入一个明确的计算块"。

9. **FixpipeOpt 与 UnifyStoreBlock 的分工？**
   流水线顺序上 FixpipeOpt 在前（L69）、UnifyStoreBlock 在后（L73）。FixpipeOpt 把 fixpipe 链标成 **CUBE**，这些 store 因此不再是 VECTOR，被 UnifyStoreBlock 的候选过滤（`getOpCoreType == VECTOR_ONLY`）排除。**剩下的普通 Vector store** 才由 UnifyStoreBlock 并入 Vector 生产者——两个 pass 各管一类，互不重叠。

10. **ExpSubfPattern 为什么不做环检测？**
    exp/extf 是 subf 的**直接下游/上游**，方向单一：把 exp（subf 的用户）和 extf（subf 的操作数）并入 subf 块不会引入反向边，天然无环。环检测（`willCreateCycle`）主要防"跨块依赖成环"，这里所有 op 都是 subf 的邻接节点，无需防御。**环检测的成本也值得省**（每模式一次全图 DFS）。
