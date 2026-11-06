# DynamicCVPipeline 子 Pass 专题：load→exp→broadcast 跨块下沉（06）

> 覆盖 `ComputeBlockOptPass` 阶段 B 中 `ExpLoadPatternPass`（[ComputeBlockOptPass.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp) L59）
>
> 源码文件：[`ExpLoadPatternPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/ExpLoadPatternPass.cpp)（231 行）
>
> 流水线位置：阶段 B 中 `MergeVectorIfBlock`（L55）+ 两次 Reorder 之后、`UBUsageOpt`（L60）之前。**main-dev 新增**。

---

## 一、这个 Pass 在做什么

`ExpLoadPatternPass` 是阶段 B（UB 优化）的一个**特化模式下沉**优化。它的目标是匹配这样一条 VECTOR 数据链：

```
memref.alloc ─► memref.copy ─► to_tensor ─► exp/exp2 ─► broadcast ─► mulf(%arg, …)
```

这条链里 `alloc → copy → to_tensor → exp → broadcast` 整体属于 block_id = X（一个 VECTOR 块），而链尾的 `mulf` 属于另一个 block_id = Y（另一个 VECTOR 块）。如果保持原状，X 块产出的 `broadcast` 结果要跨块传给 Y 块的 `mulf`，多一次跨块（核内存取）往返。

该 Pass 把这条链整体（连同其标量依赖闭包）**搬到 Y 块**（`mulf` 所在的块），让 `exp → broadcast → mulf` 在同一块内生产消费，省掉一次跨块传输；同时保留 X 块内的原链供 X 块其它使用者使用（通过标量克隆破坏跨块依赖）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ExpLoadPattern` | 该 Pass 匹配的结构体：`broadcastOp / expOp / toTensorOp / copyOp / allocOp / mulfOp` + 原 blockId + 目标 newBlockId |
| `matchExpLoadPattern` | 逐步收紧的匹配器：核心类型必须是 VECTOR、`mulf` 另一操作数是 block 参数、各环节同 blockId、`mulf` 在另一 blockId |
| `isExpLike` | `math.exp` 或 `math.exp2` 二选一 |
| `ScalarClosure`（复用） | 复用 `SplitIf` 目录的标量闭包工具，收集整条链依赖的外部标量 |
| `cloneScalarOpsForCrossBlockUses` | ComputeBlockOpt Common：克隆"被 pattern 外 op 使用"的标量生产者，破坏移动后可能出现的跨块依赖环 |
| `willCreateCycle` | ComputeBlockOpt Common：临时代入目标 blockId，走 SSA+内存依赖验证并块不会成环，且**rollback 临时 id** |
| `updateBlockIdWithInner` | 把 op 及其子 region 内 op 统一改为目标 blockId |

---

## 三、逐行讲解

### 3.1 模式匹配 `matchExpLoadPattern`（L71-L149）

逐步收紧的匹配链（任何一步失败即放弃）：

1. **broadcast 必须是 VECTOR_ONLY**（L74-L77）：`getOpCoreType(broadcastOp) != VECTOR_ONLY` → 放弃；
2. **broadcast 有 blockId**（L78-L81）：`getBlockIdByOp(broadcastOp) == -1` → 放弃；
3. **broadcast 的 dpsInput[0] 是 exp/exp2**（L83-L90）：`isExpLike(expOp)` 且 `expOp` 同 blockId 且 `hasOneUse`；
4. **exp 的输入是 to_tensor**（L92-L97）：`to_tensor` 同 blockId 且 `hasOneUse`；
5. **to_tensor 的 buffer 是 alloc**（L99-L102）：`alloc` 同 blockId；
6. **alloc 有一个 copy 写入**（L104-L115）：在 alloc 的 user 里找 `memref.copy`，要求其 target == alloc、同 blockId、且与 broadcast 同 block；
7. **broadcast 只有一个用户且是 mulf**（L117-L123）：`broadcast->hasOneUse()` 且唯一用户是 `arith::MulFOp`；
8. **mulf 在另一个 blockId**（L125-L131）：`newBlockId != blockId` 且 mulf 是 VECTOR_ONLY；
9. **mulf 的另一操作数是 BlockArgument**（L133-L138）：`mulf(broadcastResult, %arg)` 其中非 broadcast 那侧必须是 block 参数。

全部满足才填入 `ExpLoadPattern` 返回 true。

**设计意图**：链要"干净"——每一步都 `hasOneUse`/唯一 copy，保证整条链没有旁路使用者，搬走不会破坏其它消费；`mulf` 的另一侧是 block 参数则说明 `mulf` 是"独立消费这个 broadcast 的稳定模式"（没有额外的跨块复杂依赖）。

### 3.2 Pass 入口 `runOnOperation`（L167-L220）

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) return;

  auto &aa = getAnalysis<AliasAnalysis>();
  MemoryDependenceGraph memGraph(module, aa);
  ComputeBlockIdManager bm(module);

  // Phase 1: 收集所有匹配
  SmallVector<ExpLoadPattern> patterns;
  module.walk([&](linalg::BroadcastOp broadcastOp) {
    ExpLoadPattern info;
    if (matchExpLoadPattern(broadcastOp, bm, info))
      patterns.push_back(info);
  });

  // Phase 2: 逐个应用
  for (auto &info : patterns) {
    // 防陈旧：匹配后 block id 可能被前面的 pattern 改写
    if (bm.getBlockIdByOp(info.broadcastOp) != info.blockId) continue;

    SmallVector<Operation *> coreOps = {info.broadcastOp, info.expOp,
                                        info.toTensorOp, info.copyOp,
                                        info.allocOp};

    // 收集外部标量依赖闭包
    SplitIf::ScalarClosure closure{info.broadcastOp->getBlock(), coreOps, false};
    closure.collect();

    SetVector<Operation *> matchedOps;
    for (auto *op : coreOps) matchedOps.insert(op);
    for (Operation *op : closure.scalarOps) matchedOps.insert(op);

    // 克隆被 pattern 外 op 使用的标量，破坏跨块依赖
    cloneScalarOpsForCrossBlockUses(bm, matchedOps, info.newBlockId);

    // 成环检测，通过才真正搬块
    if (!willCreateCycle(matchedOps.getArrayRef(), memGraph,
                         info.newBlockId, bm)) {
      for (Operation *op : matchedOps)
        bm.updateBlockIdWithInner(op, info.newBlockId);
    }
  }
}
```

**要点**：
- **两阶段**：先 `module.walk` 收集全部匹配，再统一应用——避免边 walk 边改 IR 导致的迭代器失效与匹配错乱。
- **防陈旧（stale）**：应用阶段先复查 `broadcastOp` 的 blockId 是否仍是 `info.blockId`（前面的 pattern 可能已经搬过它），变了就跳过。
- **标量闭包**：`coreOps` 只是链上 5 个 op；`ScalarClosure`（`includeParent=false`）额外收集其依赖的外部标量，共同作为"搬移单元"。
- **跨块标量克隆**：`cloneScalarOpsForCrossBlockUses` 只克隆被"pattern 外且原块外"使用者消费的标量，重定向这些外部 use 到克隆体，原标量留给原块内部使用——避免搬块后出现跨块标量依赖环。
- **成环门控**：`willCreateCycle` 临时代入 newBlockId 走 SSA+内存依赖验环（内部会 rollback），只有不成环才真正 `updateBlockIdWithInner` 搬家。**这是"安全搬家"的最后一道闸**。

---

## 四、流程总结

```
ExpLoadPatternPass
├─ Phase 1：module.walk 逐个 broadcast 调 matchExpLoadPattern
│            （9 步收紧：VECTOR / 有 id / exp / to_tensor / alloc /
│             copy / 唯一 mulf / mulf 另一 blockId / 另侧 block 参数）
└─ Phase 2：逐个应用
     ├─ 复查 blockId 防陈旧
     ├─ ScalarClosure 收集标量闭包 → matchedOps
     ├─ cloneScalarOpsForCrossBlockUses（破坏跨块标量依赖）
     └─ willCreateCycle 验环 → 通过则 updateBlockIdWithInner 搬家
```

---

## 五、面试要点

1. **`ExpLoadPattern` 在优化什么？** `alloc→copy→to_tensor→exp→broadcast` 这条 VECTOR 链整体在 X 块，但链尾 `mulf` 在另一 VECTOR 块 Y。把链整体搬到 Y，让 exp/broadcast/mulf 同块生产消费，省一次跨块（核内存取）传输。

2. **为什么匹配条件这么严（每步都要 `hasOneUse`/唯一 copy）？** 搬移的前提是"链无旁路"。若 exp 或 broadcast 还有别的使用者，搬走会破坏其它消费关系。严匹配保证"整条链是单消费者链"，搬移安全、语义不变。

3. **为什么要求 `mulf` 的另一操作数是 BlockArgument？** 这保证 `mulf` 是"稳定消费该 broadcast"的独立模式（另一侧是循环/函数参数，不引入额外跨块复杂依赖）。反过来，若另一侧也来自块内复杂计算，搬移收益和安全性都不确定。

4. **`cloneScalarOpsForCrossBlockUses` 的必要性？** 链上 op 可能依赖标量（index、i1 等），搬块后这些标量生产者若留在原块，会形成跨块标量依赖（标量不能走 tensor 链）。若有 pattern 外使用者消费这些标量，则克隆一份重定向外部 use，原标量留给原块——既破坏环，又不破坏原有消费者。

5. **`willCreateCycle` 为什么是最后一道闸？** 搬块本质是"把 op 从 A 块改到 B 块"，可能让 B 块内部、或 A↔B 之间出现 SSA/内存依赖环。`willCreateCycle` 临时代入目标 id 验证整块 DAG 无环，通过才真正 `updateBlockIdWithInner`，且会把临时 id rollback 保证 blockIdManager 状态一致。

6. **这个 Pass 与 `MergeComputeBlock` 的 `kMergeComputeBlockApplied` 门控思路有何异同？** 两者都是"激进变换 + 安全门控"：`MergeComputeBlock` 用白名单+缓冲数量门控是否运行，`ExpLoadPattern` 用严匹配 + `willCreateCycle` 门控单个 pattern 是否可搬。共同点是"能变就变，但有风险就跳"。