# DynamicCVPipeline 子 Pass 逐行讲解：PlanComputeBlockPass

> 源码文件：
> - 主 Pass：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock.cpp)
> - 公共工具：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/Common.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/Common.cpp)
> - Block ID 管理器：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.cpp)
>
> 流水线位置：`AddDynamicCVPipelinePass::addPasses` 中 **第 3 个 pass**（`StandardizeOp` 之后、`ComputeBlockOpt` 之前）
>
> 子 Pass 专题（同目录）：
> - `02_OpClassifier`：OpClassifierPass 九步分类
> - `03_PlanCubeBlock`：CUBE 分块
> - `04_PlanVectorBlock`：VECTOR 分块
> - `05_ReorderOpsByBlockId`：按 block id 重排

---

## 一、这个 Pass 在做什么

`PlanComputeBlock` 是 CV 流水线的**核心编排 pass**：它回答两个问题——**"每个 op 应该在哪颗核上执行"**（CUBE / VECTOR）和 **"哪些 op 应该组成一个计算块（compute block）"**（block id）。后续所有 Pass（`ComputeBlockOpt` 合并、`SplitDataflow` 拆分、`AllocMultiCache` 分配缓冲、`AddControlFlowCondition` 插入控制流）都建立在 block id 之上。

主 Pass 本身只是一个编排壳，按固定顺序调度 4 个子 Pass：

```cpp
void PlanComputeBlockPass::runOnOperation() {          // PlanComputeBlock.cpp L44-L75
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; } // fallback 兜底

  OpPassManager pm(module.getOperationName());

  pm.addPass(createOpClassifierPass());       // Step 1: 分类 op → CUBE / VECTOR / CUBE_AND_VECTOR
  pm.addPass(createPlanCubeBlockPass());      // Step 2: 给 CUBE op 分配 block id
  pm.addPass(createPlanVectorBlockPass());    // Step 3: 给 VECTOR op 分配 block id
  pm.addPass(createReorderOpsByBlockIdPass());// Step 4: 按 block id 分组重排 op 顺序

  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }
}
```

四个子 pass 的分工：

| 子 Pass | 输入 | 输出 |
|---|---|---|
| OpClassifierPass | 标准化后的 IR（无 block id） | 每个 op 的 `ssbuffer.core_type`（CUBE/VECTOR），同步 op 获得独立 block id |
| PlanCubeBlockPass | core_type 已标注 | 所有 CUBE 系 op 获得 block id |
| PlanVectorBlockPass | CUBE 已有 id | 所有 VECTOR 系 op 获得 block id |
| ReorderOpsByBlockIdPass | 全部 op 有 block id | 按 block id 分组拓扑重排后的 IR |

**为什么这么排**：分类必须先于分块（不知道 op 是 CUBE 还是 VECTOR 就无法决定给谁发 id）；分块必须先于重排（重排按 block id 分组）。CUBE 先分块是因为 CUBE 数量少且以 matmul 为核心种子，先给 CUBE 定 id 后，VECTOR 分块才能引用"哪些 op 已属于 CUBE 块"来做依赖边界判断（见 04 篇的 `findOpsAdjacentToCube`）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ssbuffer.core_type` | OpClassifier 写给每个 op 的属性，取值 `"CUBE"` / `"VECTOR"` / `"CUBE_AND_VECTOR"`。后续 pass 靠它判断 op 归属 |
| `ssbuffer.block_id`（`kBlockId`） | 计算块唯一编号。**同一编号的 op 属于同一个计算块**，会被放到同一个核、同一次执行窗口 |
| `ComputeBlockIdManager` | 统一管理 `op → blockId` 双向映射的容器，同时是 id 分配器（`getNextId` 单调递增） |
| `-1` 特殊 id | 表示"未分配"；`getBlockIdByOp` 对无 id 的 op 返回 -1 |
| `SyncWall` | 按源码顺序为 block 内的同步 op（`gpu.barrier` / `hivm.sync_block_*`）建"墙"，提供 `segmentOf`（分段）、`hasSyncBetween` / `sameSegment`（是否跨同步）查询。**任何 block 组都不允许跨同步栅栏** |
| `DependencyHelper` | 封装"SSA + 内存"双维依赖遍历：`forEachSource` / `forEachUser` / `forEachUserInSameBlock` 等 |
| `MemoryDependenceGraph` | 内存依赖图（基于 `AliasAnalysis`），把 `store → load` 等内存边纳入依赖分析 |
| `DependencyCycleDetector` | DFS 环检测器：给定候选组，判断把某个 op 并入后是否会产生依赖环 |
| CUBE seed | OpClassifier 通过 pattern 匹配得到的"CUBE 起泡起点"（to_tensor/transpose/fill/empty/broadcast/store 等 matmul 周边 op） |
| `kExternalSync` | 主 Pass 给每个同步 op 打的标记（属性值为 1），供后续 pass 识别"外部同步点" |
| 继承（`inheritFromParent`） | 若子 block 的父 op 是纯 CUBE/VECTOR 的 scf op，子 block 直接继承父 op 的 block id，不再单独分块 |

---

## 三、逐行讲解

### 3.1 主 Pass：`PlanComputeBlock.cpp`（L44-L75）

- **L47** `hasFallbackAttr`：上游任何 pass 失败打上 fallback 属性后，本 pass 直接空跑返回——流水线整体回退标准编译。
- **L51** `OpPassManager pm(module.getOperationName())`：子 pipeline 挂在 ModuleOp 上，因为 OpClassifier 需要全局 walk 收集所有 op（分类是"全局"决策，不能逐 block 独立进行）。
- **L54-L64** 四个子 pass 顺序固定，见上文表格。
- **L66-L72** `runPipeline` 失败处理：
  - 若失败且没有 fallback 属性，打 `ERRCODE_FAILED`；
  - `runPipeline` 内部任一子 pass 已自行打过 fallback 时，保持原有错误码不动（用 `if (!hasFallbackAttr)` 保护，避免覆盖更精确的错误码）。
- **L79-L90** 注册逻辑：`registerPlanComputeBlockPasses` 同时注册主 Pass 和三个子 Pass，保证 `mlir-opt --plan-cube-block` 等命令可用。

### 3.2 公共工具：`Common.cpp`（L34-L57）

```cpp
void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const DependencyHelper &depHelper,
                                ComputeBlockIdManager &bm) {   // L34-L46
  for (auto *op : llvm::make_pointer_range(block->getOperations())) {
    indegree[op] = 0;
    depHelper.forEachSource(op, [&](Operation *source) {
      // 只统计"同 block 内、且不属于同一计算块"的源 → 跨块依赖才计入入度
      if (source->getBlock() == block && !bm.isSameBlock(source, op)) {
        indegree[op]++;
      }
    });
  }
}
```

- 拓扑排序的入度初始化。关键点：`!bm.isSameBlock(source, op)` —— **同一计算块内的依赖不算入度**。因为分块后块内 op 是"一起调度"的，只有跨块的依赖才是真正的调度约束。这在 CUBE/VECTOR 两个拓扑分块中都被复用。
- `getAncestorInBlock`（L48-L57）：沿 `getParentOp` 向上找"所在 block 等于目标 block"的祖先 op。用途：嵌套在 scf/linalg 内部的 op，映射到其所属 block 层面的代表 op（ReorderOpsByBlockId 里 `resolveToBlockOp` 用到）。

### 3.3 Block ID 管理器：`ComputeBlockIdManager.cpp`（核心基础设施）

#### 3.3.1 构造：扫描已有 id（L40-L56）

```cpp
ComputeBlockIdManager::ComputeBlockIdManager(Operation *root) {   // L40-L56
  root->walk([&](Operation *op) {
    if (auto blockIdAttr = op->getAttrOfType<IntegerAttr>(kBlockId)) {
      auto blockId = blockIdAttr.getInt();
      if (blockId <= 0) { return; }          // 忽略非法 id
      opToBlockId[op] = blockId;
      blockIdToOps[blockId].push_back(op);
      cntComputeBlockId = std::max(cntComputeBlockId, blockId);
    }
  });
  cntComputeBlockId++; // 保证下一个新 id 唯一且不冲突
}
```

- 每次分块 pass 都会新建一个管理器，先扫描 IR 里已有的 `ssbuffer.block_id`（例如 OpClassifier 给同步 op 发的 id），把计数器推到"已用最大值 + 1"，保证 `getNextId()` 永不与已有 id 冲突。
- 这个机制让 4 个子 pass 共享同一个 id 空间（CUBE 和 VECTOR 的 id 是统一编号的，头文件注释 `the class is to promise CUBEID and VECTORID are unified`）。

#### 3.3.2 查询类 API

- `isSameBlock(a, b)`（L71-L76）：两者 block id 相同且都不是 -1 → 同块。
- `isWholeCubeReady(seedOp, indegree)`（L58-L69）：**seedOp 所在块的所有 op 入度都为 0** 才算"整块就绪"。这是"按块调度"的关键语义——CUBE 块整体作为一个调度单元。
- `getOpsInSameBlock(op)`（L186-L202）：返回与 op 共享 block id **且在同一 MLIR block 内**的 op 集合。同 id 但不同 block（如嵌套 scf 同名块）会被过滤掉。
- `getBlockIdByOp(op)`（L213-L215）：返回 -1 表示未分配。

#### 3.3.3 分配/更新类 API

```cpp
llvm::LogicalResult ComputeBlockIdManager::markAndRecord(Operation *op, int blockId) {  // L217-L234
  op->setAttr(kBlockId, ...);          // 直接写 IR 属性
  auto itOld = opToBlockId.find(op);
  if (itOld != opToBlockId.end() && itOld->second != -1) {
    llvm::errs() << "Error: Operation already has a block id. ...";
    return llvm::failure();            // 重复标记 → 报错（防呆）
  }
  opToBlockId[op] = blockId;
  blockIdToOps[blockId].push_back(op);
  return llvm::success();
}
```

- `markOpBlockId(op)`（L236-L239）：`getNextId()` + `markAndRecord`，给单个 op 发全新 id。
- `markOpsWithNewId(ops)`（L241-L253）：一批 op 共用一个新 id（分块的核心入口，CUBE/VECTOR 分块都调用它）。
- `updateBlockId(op, blockId)`（L117-L144）：**强制覆盖**已有 id（重标记），并同步维护双向映射（从旧 id 集合移除、插入新 id 集合）。`blockId == -1` 时删除属性（解除标记）。
- `updateBlockIdWithInner(parentOp, targetId)`（L80-L115）：把父 op 连同其子 block 内的 op 一起更新为目标 id。保护规则：
  - 父 op 无 id → 只更新父 op；
  - 父 op 是 linalg 方言 op → 只更新父 op（linalg 内部 op 不应有 block id）；
  - 子 block 内存在 id 与父不一致的 op → 放弃（不强行覆盖，避免破坏已有分组）；
  - **同步 op 永远不被并入**（`if (isSyncOp(op)) return;`）——同步点必须保持独立 id，否则 ReorderOpsByBlockId 造的 fence 会失效。
- `forgetOp(op)`（L146-L165）：从双向映射中移除（未被记录则直接返回）。

#### 3.3.4 继承机制（L255-L290）

```cpp
bool ComputeBlockIdManager::shouldInheritFromParent(Block *block, CoreType requiredCoreType) const {
  auto *parentOp = block->getParentOp();
  if (!parentOp || !isScfOp(parentOp) ||
      getCoreTypeOfSimpleOpOrCf(parentOp) != requiredCoreType) {
    return false;
  }
  auto blockIdOpt = getBlockIdByOpOpt(parentOp);
  return blockIdOpt.has_value();
}
```

- 当 scf 父 op（for/while）整体被标为纯 CUBE（或纯 VECTOR）且已有 block id 时，其所有子 block 直接 `inheritFromParent`（L267-L290）继承父 id，**不再走独立分块算法**。
- 语义：父 op 作为整体是一个计算块，内部 op 共享同一 id。这保证后续 `ReorderOpsByBlockId` 把整个 scf 结构作为一个组移动，不破坏内部顺序。
- 注意 `inheritFromParent` 只标记 `*block` 的直接 op（注释明确说明不再向嵌套 block 递归——walk 本身会逐层访问，重复标记同 id 是无害的）。

---

## 四、算法流程总结

```
PlanComputeBlockPass（编排壳）
├─ Step 1  OpClassifierPass：9 步给每个 op 标 core_type，同步 op 先发独立 block id
│          （详见 02 篇：pattern 找种子 → CUBE 向上 BFS → loader 循环渗透
│           → 剩余默认 VECTOR → VECTOR 向上 BFS → CUBE_AND_VECTOR 分裂
│           → yield 类型标注 → stamp 到 IR）
├─ Step 2  PlanCubeBlockPass：CUBE op 分块
│          Phase 1  matchSeed + SeedRegionPlanner（BFS 融合，cycle 检测 + 不跨同步）
│          Phase 2  TopologicalPartitionPlanner（拓扑处理剩余 CUBE，按同步分段发 id）
│          fuseMarkOpToDef：annotation.mark 并入其定义 op 的块
├─ Step 3  PlanVectorBlockPass：VECTOR op 分块
│          拓扑入度 → 融合（bypass 非可融合 op）→ 正反两遍 refine 切割
│          → 按同步分段发 id
├─ Step 4  ReorderOpsByBlockIdPass：按 block id 分组重排
│          构建 op 级 DAG → group 级 DAG → Kahn 拓扑 → 组内重排（VECTOR 块 sink store）
│          → moveBefore 重写 IR 顺序
└─ 任意一步失败 → setFallbackAttr(ERRCODE_FAILED) → 整体回退标准编译
```

**不变式（invariant）**：任意计算块（block id 组）绝不跨同步栅栏。这一约束在 OpClassifier（同步 op 独享 id）、CUBE 分块（`tryAddToGroup` 检查 `hasSyncBetween`、分段发 id）、VECTOR 分块（`segmentOf` 分段）、Reorder（sync fence 校验）四层反复强化。

---

## 五、面试要点

1. **PlanComputeBlock 在整个流水线中扮演什么角色？**
   它是"分块决策器"。前面的 `StandardizeOp` 负责把 IR 形态摆正，`PlanComputeBlock` 决定**每个 op 去哪颗核、和谁一组**（block id），后面的 `SplitDataflow`/`AllocMultiCache` 都在 block id 的粒度上做核间数据流拆分与缓冲分配。可以说它是 CV 流水线能否正确拆分的前提。

2. **为什么分类必须是全局的（ModuleOp 级），而分块是逐 block 的？**
   分类依赖 use-def 全链传播（一个 op 的类型由它上游的种子决定），天然是全局问题；分块则发生在每个基本块内部（同 block 的 op 才可能共组）。所以 OpClassifier 的 pass 锚定 ModuleOp，而 CUBE/VECTOR 分块 pass 用 `walk(Block*)` 逐块处理。

3. **`ComputeBlockIdManager` 为什么是"统一 id 空间"？**
   CUBE 块和 VECTOR 块用同一个计数器编号（CUBE 先发 1,2,3…，VECTOR 接着发），这样 `blockIdToOps` 能查回任意块（不管 CUBE/VECTOR）的所有成员。后续 ReorderOpsByBlockId 的 group 级 DAG 需要同时处理两类块，统一编号是前提。

4. **"同步 op 独享 block id"背后的设计动机？**
   同步栅栏（barrier/sync_block）在 IR 里是"墙"：墙两侧的 op 必须保持相对顺序。如果同步 op 被并入某个计算块，重排时同步点会跟着块一起移动，栅栏语义就丢了。所以：OpClassifier 给同步 op 发唯一 id → 分块时"块不跨同步" → 重排时同步 op 保持独立组。三层机制保证同一不变式。

5. **`isWholeCubeReady` 与普通拓扑"入度为 0 即可调度"有什么不同？**
   普通拓扑按 op 粒度判断；`isWholeCubeReady` 按**块**粒度判断——块内所有 op 入度均为 0 才认为该块可调度。这是"CUBE 块整体进、整体出"语义的体现：一个块要么整体不执行，要么整体执行，不能把块拆开调度。

6. **"继承"（inheritFromParent）解决了什么问题？**
   当整个 scf.for/while 被标为纯 CUBE（或纯 VECTOR）时，块内部再做独立分块是浪费且危险（可能把同一逻辑拆散）。继承让子 block 直接复用父 op 的 id，保证"整环一体"，也减少 id 数量、降低后续重排/传输的复杂度。

7. **fallback 在这里为什么用 `if (!hasFallbackAttr)` 保护？**
   `runPipeline` 失败时，具体失败点可能已打了更精确的 fallback 码（如 OpClassifier 在某个步骤打 `ERRCODE_FAILED`）。主 Pass 若无条件覆盖，会丢失失败来源信息；加了保护后只在"子 pass 还没标记"时才补标，保留最具体的错误语义。

8. **四个子 pass 顺序能否交换？为什么？**
   不能。①分类必须在分块前（分块依赖 core_type）；②CUBE 分块必须在 VECTOR 分块前（VECTOR 分块需要知道哪些 op 已在 CUBE 块中，以便把"紧邻 CUBE 的 VECTOR 计算"正确切割）；③两者都必须在重排前（重排以 block id 为分组依据）。这个顺序本身就是整个流水线设计意图的浓缩。
