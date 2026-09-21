# DynamicCVPipeline 子 Pass 专题：UB 与加载侧优化（02）

> 覆盖 `ComputeBlockOptPass` 阶段 A + 阶段 B 的 7 个子 Pass：
> `UnifyAllocBlock` / `UBUsageOpt` / `BroadcastUBOpt` / `PosMaskPattern` / `MergeSameSourceAxis` / `MergeSmallBlock` / `RelocateMemrefDecl`
>
> 源码文件：
> - [`UnifyAllocBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/UnifyAllocBlockPass.cpp)
> - [`UBUsageOptPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/UBUsageOptPass.cpp)
> - [`BroadcastUBOptPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/BroadcastUBOptPass.cpp)
> - [`PosMaskPatternPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/PosMaskPatternPass.cpp)
> - [`MergeSameSourceAxisPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeSameSourceAxisPass.cpp)
> - [`MergeSmallBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeSmallBlockPass.cpp)
> - [`RelocateMemrefDeclPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/RelocateMemrefDeclPass.cpp)
>
> 流水线位置：`ComputeBlockOptPass`（01 篇）中阶段 A（L51-L52）与阶段 B（L58-L64），外加阶段 D 收尾的 `RelocateMemrefDecl`（L85）

---

## 一、这个专题在做什么

`PlanComputeBlock` 的分块是"以正确性为先"的：**每个 op 都要有归属，块内无环，块不跨同步栅栏**。代价是块的边界往往落在"数据结构不自然"的位置——例如：

- `memref.alloc` 被分配在某个块，但真正使用它的 load/subview 在另一个块；
- broadcast / pos-mask 这类**产生小数据**的 op 孤零零待在一个块，消费者却在别处；
- 多个消费者块共享同一个数据源，导致重复的跨块缓冲；
- 某块只有 1-3 个 op，块间往返开销远大于计算本身。

本专题的 7 个子 Pass 分三条主线解决这些问题：

| 主线 | 子 Pass | 核心动作 |
|---|---|---|
| **A. 内存声明归属** | `UnifyAllocBlock` | 把 `memref.alloc`（及其配套的 `scf.if(linalg.fill)` 初始化、`memref.subview` 视图）整体并入消费它们的块 |
| **B. UB 依赖收紧** | `UBUsageOpt` | **核心 pass**：把"共用 UB 且生命周期重叠"的 Vector 块按最小 UB 依赖位置切分开 |
| | `BroadcastUBOpt` / `PosMaskPattern` | 把产生小数据（broadcast / pos-mask 模式）的 op 下沉到用户块，就近消费省 UB |
| | `MergeSameSourceAxis` | 多个 Vector 消费者块的数据流在一个共同汇聚点收敛时，把它们并入数据源块 |
| | `MergeSmallBlock` | ≤3 个计算 op 的小块并入邻居块（**此 pass 在流水线里跑两轮**） |
| **C. 声明跨同步搬家** | `RelocateMemrefDecl` | 跨同步 op 的 memref 声明连同整个依赖闭包搬到消费者块 |

所有子 Pass 的**唯一本质操作都是改写 `ssbuffer.block_id`**（偶尔连同 `moveBefore` 物理搬移），不增删任何计算语义。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| UB（Unified Buffer） | Vector 核的片上统一缓冲。Vector 计算的数据必经 UB。**UB 占用 = 块间边上的数据体积**；生命周期重叠的块如果合用 UB 会超容量 |
| `getValueSizeInBytes` 权重约定 | Tensor/Vector → 数据字节数；Memref → `MAX_EDGE_SIZE`（`1<<30`，标记"此处不可切分"）；Index/标量 → 0（不占 UB）。这套约定是 UB 优化图建模的基础 |
| CUBE 块收缩（shrink） | `buildUBUsageGraph` 中同一 block_id 的 CUBE op 合并为一个图节点——CUBE 内部依赖对 UB 切分无意义，只在块边界计算 |
| 同步栅栏（SyncOp） | `RelocateMemrefDecl` 关心的对象：跨同步的 memref 声明在后续 SplitDataflow 降栅栏时会被破坏，必须提前搬家 |
| `kSubBlockId`（`setSubBlockId`） | 块合并时给"被并入块/目标块"打原始 id 标记，记录合并历史供下游 pass 使用 |
| `kMergeSmallBlockFirstRunDone` | Module 级属性：标记 `MergeSmallBlock` 是否第一轮跑过（该 pass 被调度两次，sitofp-subf-mulf 模式只在第二轮合并） |
| `firstRun` / 程序顺序 | 小块"向上游还是下游并"需要仲裁：先用 IR 程序顺序（`id2order`）分上下，再让策略细化选择 |

---

## 三、逐行讲解

### 3.1 UnifyAllocBlockPass（UnifyAllocBlockPass.cpp）

**做什么**：把 `memref.alloc` 的归属统一到"真正消费它的块"。典型场景是 alloc 被一个 `scf.if` 里的 `linalg.fill` 初始化、随后被 `memref.subview` 切视图后由 `memref.copy` 读取——alloc/if/fill/subview 必须跟着 copy 的块走，否则声明和首次使用跨块。

**Pass 入口（L333-L357）**：

```cpp
void UnifyAllocBlockPass::runOnOperation() override {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }        // L336-L338 fallback 兜底

  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(module, aa);     // L341-L342 内存依赖图
  auto bm = CVPipeline::ComputeBlockIdManager(module);

  llvm::SmallVector<memref::AllocOp> allocOps;
  module.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });  // 收集全部 alloc

  for (memref::AllocOp allocOp : allocOps) {
    if (failed(tryUnifyForAlloc(allocOp, memGraph, bm))) {    // L350 失败 → fallback
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }
}
```

注意 **L350 失败（failure）才 fallback**，而 `tryUnifyForAlloc` 内部多数"不符合模式"的返回是 `success()`（静默跳过）。只有 `getCommonBlockId` 发现多个不同块、或 `willCreateCycle` 发现环才真正报错。**设计意图**：模式不匹配是常态（跳过），结构矛盾才是错误（fallback）。

**核心逻辑 `tryUnifyForAlloc`（L261-L315）**，六步：

```cpp
static LogicalResult
tryUnifyForAlloc(memref::AllocOp allocOp,
                 const CVPipeline::MemoryDependenceGraph &memGraph,
                 CVPipeline::ComputeBlockIdManager &bm) {
  // Step1: 收集直接用户（排除 linalg.fill）                 L266-L271
  Value allocResult = allocOp.getResult();
  SmallVector<Operation *> directUsers = collectDirectUsers(allocResult);
  if (directUsers.empty()) { return success(); }

  // Step2: 在 scf.if（无 else、fill 在 then region）里找 linalg.fill  L274-L279
  FillInfo fillInfo = findFillOpInSCFIf(allocResult);
  if (!fillInfo.fillOp) { return success(); }

  // Step3: 所有直接用户必须同属一个 block_id               L282-L287
  int targetBlockId;
  if (failed(getCommonBlockId(directUsers, targetBlockId))) { return failure(); }

  // Step4: scf.if 的 then region 只能有 fill 这一个 op       L290-L293
  if (hasOtherOpsInIf(fillInfo)) { return success(); }

  // Step5: 收集统一对象 = alloc + fill + if + 直接用户 + 视图链  L296-L302
  SmallVector<Operation *> coreOps = {allocOp, fillInfo.fillOp, fillInfo.parentIf};
  coreOps.append(directUsers);
  coreOps.append(collectSourceViewChainOps(directUsers));

  // Step6: 环检测 + 改写 block_id                          L305-L311
  if (CVPipeline::willCreateCycle(coreOps, memGraph, targetBlockId, bm)) {
    return failure();
  }
  for (auto *op : coreOps) { bm.updateBlockId(op, targetBlockId); }
  return success();
}
```

几个关键 helper：

- **`collectDirectUsers`（L70-L78）**：遍历 `allocResult.getUsers()`，**排除 `linalg::FillOp`**。原因（源码注释 L65-L68）：`linalg.fill` 是 DestinationStyleOp，`outs(%alloc)` 是内存位置而非 SSA 依赖，所以 fill **不会**出现在 `getUsers()` 里——排除它是防御性的。
- **`findFillOpInSCFIf`（L144-L173）**：找同时满足 4 个条件的 fill：① 用该 alloc 作 `getDpsInits()[0]`；② 在 `scf.if` 内；③ if **没有 else region**；④ fill 位于 **then region 的第一个 block**。这是"if 就是给 alloc 做初始化"的典型三明治结构。
- **`getCommonBlockId`（L95-L131）**：所有直接用户的 block_id 必须相同。**subview 特判**：若直接用户含 `ViewLikeOpInterface`（如 subview），则穿透到 `memref.copy`，取 copy 的 block_id（而非 subview 自己的）。若穿透出多个不同 copy 的块 → `failure()`。**意图**：subview 是纯视图、没有独立归属，真正决定归属的是读它的 copy。
- **`hasOtherOpsInIf`（L187-L199）**：`without_terminator()` 数 then region 里的 op，>1 表示 if 里混了别的逻辑 → 放弃统一（把整个 if 划进某块会把无关逻辑也带走，错误）。
- **`collectSourceViewChainOps`（L230-L251）+ `traceViewChain`（L203-L215）**：从 copy 的 source 沿 `ViewLikeOpInterface` / `tensor.extract_slice` 链回溯（如 `subview → reinterpret_cast → extract_slice`），收集同 block 的链上 op，一并纳入 `coreOps` 统一搬家。**意图**：声明、初始化、视图、读取要"同生共死"。

### 3.2 UBUsageOptPass（UBUsageOptPass.cpp）

**做什么**：主 pass 注释点名的核心——"find the smallest UB dependency location and divide the computation blocks"（找最小 UB 依赖位置来切分计算块）。当两个 Vector 块通过一个 value 共享 UB 时，若该 value 的生命周期跨越块边界且与另一块的占用重叠，就可能超 UB 容量。本 pass 把"必须分开"的边界找出来：**在边权重最小（UB 占用最少）的位置切断**。

**Step 0：权重约定 `getValueSizeInBytes`（L114-L152）**

| 类型 | 权重 | 含义 |
|---|---|---|
| RankedTensor（静态 shape） | 元素数 × 元素字节数 | UB 真实占用 |
| RankedTensor（动态/负数维度） | `MAX_EDGE_SIZE` / 1 | 不可精确估计 |
| Memref | `MAX_EDGE_SIZE`（`1<<30`） | **此处禁止切分**（memref 只是声明，不占 UB；跨块切分会导致声明脱离使用） |
| Vector | 元素数 × 字节数 | 真实占用 |
| Index/其他 | 0 | 标量不占 UB |

**Step 1：建加权依赖图 `buildUBUsageGraph`（L154-L323）**

```cpp
auto getOrCreateNodeId = [&](Operation *op) -> int {          // L167-L196
  ...
  bool canShrink = (coreType == cubeCoreType && blockId != -1);
  if (canShrink) {                                            // CUBE 块收缩
    auto it = cubeBlockId2nodeId.find(blockId);
    if (it != cubeBlockId2nodeId.end()) {
      op2nodeId[op] = it->second;                             // 同一 CUBE 块 → 同一节点
      return it->second;
    }
  }
  ... // 新建节点
  if (canShrink) { cubeBlockId2nodeId[blockId] = nodeId; }
  return nodeId;
};
```

- **CUBE 收缩（L173-L195）**：同 block_id 的 CUBE op 全部折叠成一个图节点。CUBE 内部依赖与 Vector 的 UB 切分无关，只在块边界（CUBE→Vector 的边）有意义。这是图规模的第一个剪枝。
- **yield 参数标注（L219-L235）**：若块 terminator 是 `scf::YieldOp`，把 yield 的第 argIdx 个操作数对应的定义 op 标为 `nodeArgs = argIdx`——这是"块输出参数"，后续边权重 ×2（L283-L285）因为它要跨迭代存活。
- **SSA 边（L239-L291）**：对 block 内每个 op 的每个 operand，若定义者在本 block 内（`getAncestorInBlock`），则加一条 `srcNode → dstNode` 边，权重 = 值大小。两处**防自环**：
  - L245-L250：srcInBlock == blockOp（内部 op 用父 op 参数）→ 跳过；
  - L277-L279：**同 CUBE 块的 src/dst → 跳过**（CUBE 内部依赖不参与 UB 切分）。
- **内存边（L293-L320）**：对 `memGraph.getExecBefore(op)` 的每个内存前驱加边，**权重 0**（内存依赖只约束顺序，不占 UB）。同样跳过 CUBE 块内自环。
- **BlockArgument 回溯（L251-L268）**：operand 是 block 参数时，通过 yield 找到它对应的定义 op（`offset = numArgs - numYieldOperands`，for op 偏移 1、while op 偏移 0——**for 有 induction variable 参数，while 没有**），并标记 `fromArgEdge`（边权重 ×2）。

**Step 2：候选收集 `collectNeedUbOpts`（L387-L423）**

只关心 **VECTOR_ONLY 块**（`srcCoreType != VECTOR_ONLY → continue`）。一个节点成为候选的条件：存在一条出边指向 **active end node**（`isActiveEndNode`）。

`isActiveEndNode`（L352-L385）判定"这个终点节点是否值得切"：
- 与源节点同 core type（都是 Vector）；(L359)
- 有 block_id（不是 `-1`，即不是 yield/return 等控制流节点）；(L362-L365)
- 与源节点**不同块**（同一块内部不是切分点）；(L366-L369)
- **防环**：endNode 的上游依赖（`findDependency`，BFS 沿 `linkIn` 反向展开，跳过源节点自身）中，若出现"属于第三个块且有入边"的节点 → 不是合法切分点（切了会造环）。(L375-L383)

**Step 3：链上最小割点 `collectRecordChange`（L467-L551）**

对每个候选 optNode：

```cpp
// 1) 找出所有 active end node 组成 activateSet（L481-L492）
// 2) 对每个 activateNode：
int64_t originUBSize = sumIncomingLinkSize(activateNode, ...);  // 当前切分点的 UB 占用
int64_t minUBSize = originUBSize;
SmallVector<int> chain = {activateNode};
// 3) 沿"唯一后继"单链扩展（L505-L515）
while (true) {
  if (!findUniqueDependentNode(curNode, ...)) break;            // 出边≠1 / 跨块 / 依赖越界 → 停
  chain.push_back(uniqueNextNode);
}
// 4) 找链上 UB 占用最小的切点（L516-L531）
for (auto i = 0; i < chain.size(); i++) {
  nowUBSize = Σ linkOut[chain[i]] 的边权重（遇到 MAX_EDGE_SIZE 直接封顶）
  if (nowUBSize < minUBSize) { bestCutPointIdx = i + 1; minUBSize = nowUBSize; }
}
// 5) 记录：切点之前的链成员 + 其依赖全部并入 optBlockId（L533-L546）
```

`findUniqueDependentNode`（L442-L465）要求"单链"：出边必须恰 1 条、后继与当前同块、后继的所有上游依赖都在当前块内——保证切分不撕裂链。

**Step 4：应用变更 `applyRecordChange`（L553-L588）**

- 按目标块分组（`blockWilladd[optBlockId]`）；
- 每组先 `willCreateCycle` 检测（**组级，不是单 op 级**——多个 node 一起迁入可能组内形成环）；
- 通过后 `bm.updateBlockIdWithInner`（**带内部 region 递归**更新）。

**Step 5：入口 `UBUsageOptimization`（L590-L635）**

```cpp
if (!(isa<scf::ForOp>(block->getParentOp()) ||
      isa<scf::WhileOp>(block->getParentOp()))) {
  return llvm::success();          // L593-L596 只处理循环体块
}
```

UB 优化只对**循环体**有意义——UB 生命周期重叠问题只在迭代/流水重叠场景出现；非循环块天然顺序执行，无需切分。`runOnOperation`（L637-L660）walk 所有 block 逐个优化。

### 3.3 BroadcastUBOptPass（BroadcastUBOptPass.cpp）

**做什么**：把 `linalg.broadcast`（广播，产生的小数据）移动到**它的全部用户所在块**，省一次跨块缓冲。

```cpp
void BroadcastUBOptPass::runOnOperation() {
  moduleOp.walk([&](linalg::BroadcastOp op) {
    if (getOpCoreType(op) != CoreType::VECTOR_ONLY) return;   // L76-L78 只处理 Vector 广播
    if (op->getUsers().empty()) return;

    // 只处理简单场景：所有用户同 block、同 block_id
    Operation *oneUser = *op->getUsers().begin();             // L84
    if (oneUser->getBlock() != op->getBlock()) return;        // L85 用户必须同 block
    int firstUserBlockId = bm.getBlockIdByOp(oneUser);
    if (firstUserBlockId == -1 || coreType(oneUser) != VECTOR_ONLY) return;  // L89-L94

    bool allUsersSameBlock = llvm::all_of(op->getUsers(), ...);// L95-L97
    if (!allUsersSameBlock) return;
    if (broadcastBlockId == firstUserBlockId) return;         // L104 已在同一块

    if (CVPipeline::willCreateCycle({op}, memGraph, firstUserBlockId, bm)) return;  // L110
    bm.updateBlockId(op, firstUserBlockId);                   // L116
  });
}
```

**约束理解**：
- 用户必须**同 block 且同 block_id**（L84-L99）：只有"用户已聚在一处"时才值得把 broadcast 沉下去；用户分散则下沉无意义（还得复制）。
- 用户必须**非控制流**（block_id ≠ -1）且 **VECTOR**（L89-L94）：控制流节点没有固定块归属；非 Vector 用户（如 CUBE 用 broadcast）下沉到 CUBE 块会改变广播的 core 归属，超出本 pass 职责。
- 下沉前 `willCreateCycle`（L110）：broadcast 移到用户块后若在用户块内产生环（用户又依赖 broadcast 的源……），则放弃。

### 3.4 PosMaskPatternPass（PosMaskPatternPass.cpp）

**做什么**：针对 attention 里常见的 **pos mask 模式**（`broadcast + cmpi(eq) + cmpi(sle) + extui + extui`）——位置掩码的计算——把整组 op 下沉到掩码用户的块。

**模式定义 `PosPattern`（L70-L80）**：一个 broadcast 结果恰好被两个 `arith::CmpIOp` 消费（一个 `eq`、一个 `sle`），每个 cmp 又恰好被一个 `arith::ExtUIOp` 消费。

**`collectPosPattern`（L90-L158）** 校验：
- broadcast 结果用户数恰为 2，且全是 `CmpIOp`（L94-L100）；
- 两个 cmp 的 predicate 必须**恰好是 eq 和 sle**（L105-L121），多一个、少一个、或是其他 predicate → 拒绝；
- 每个 cmp 结果**单用**且用户是 `ExtUIOp`（L126-L136）；
- **整组 5 个 op 必须同 block、同 block_id**（L144-L155）——模式内部已经聚拢，才谈得上整体搬迁。

**`findTargetBlock`（L160-L193）**：从 eqext/sleext 的用户里找**同 block** 的（跨 block 用户忽略），且这些用户 block_id 全部相同 → 目标块。

**`runOnOperation`（L200-L243）**：对每个匹配模式，目标块与 broadcast 块不同时，`willCreateCycle` 通过后整组 `updateBlockId`。

**设计意图**：pos mask 是一组**极小的中间计算**（位置比较 + 扩宽），单独成块会让每个 mask 值都跨块传输。整个模式下沉到掩码消费点，掩码就地产生、就地使用，与 BroadcastUBOpt 是同一思想（小数据就近）的模式特化版本。

### 3.5 MergeSameSourceAxisPass（MergeSameSourceAxisPass.cpp）

**做什么**：当某个数据源 op 有 **≥2 个 Vector 消费者块**、且这些消费者的下游在**同一个汇聚点（convergence）** 汇合时，把"汇聚点及其上游链"并入数据源块——消除"数据源在 A 块、消费链在 B/C 块"的割裂。

**Pass 描述（L251-L256，源码自带）**：
> Rewrite ssbuffer.block_id so that ≥2 vector consumers of a source op whose downstream converges at a common op end up in the same block as the source. Only VECTOR core ops are considered.

**核心：`findNearestConvergence`（L97-L154）**——BFS 找最近公共汇聚点：

```cpp
// 从所有 consumers 出发做多源 BFS，firstIdx 记录"谁先到达"
for (auto &consumer : consumers) {
  firstIdx.insert({consumer, consumerIndex});
  parent[consumer] = nullptr;
  bfsQueue.push_back({consumer, consumerIndex});
}
while (bfsQueue 非空) {
  for (Operation *user : cur->getUsers()) {
    if (getOpCoreType(user) != VECTOR_ONLY) continue;         // 只沿 Vector 向下走
    if (!isInSameRegion(user, source)) continue;
    auto it = firstIdx.find(user);
    if (it == firstIdx.end()) {
      firstIdx[user] = myIdx; parent[user] = cur;             // 首次到达，继续 BFS
      bfsQueue.push_back({user, myIdx});
    } else if (it->second != myIdx) {                          // 被第二个消费者到达 → 汇聚点！
      if (llvm::is_contained(consumers, user)) continue;      // 用户本身是消费者则跳过
      Operation *convergenceOp = user;
      // 收集 汇聚点→cons1 和 cur→cons2 两条路径 + 汇聚点的同块 Vector 尾巴
      appendPath(convergenceOp, cons1, parent, inChain, chainOps);
      appendPath(cur, cons2, parent, inChain, chainOps);
      extendChainWithConvergenceTail(convergenceOp, source, inChain, chainOps);
      return true;
    }
  }
}
```

**要点**：
- **`parent` 图记录"我由谁到达"**，`appendPath`（L54-L67）沿 parent 链回溯，收集"汇聚点 → 消费者"两条路径的所有 op（dedup 用 `inChain`）。这些 op 就是要并入源块的候选链。
- **`extendChainWithConvergenceTail`（L72-L95）**：若汇聚点自己还有**同 block 的 Vector 用户**（且同 block_id），也一并带上——防止"链头搬走了、尾巴留在原块"。
- **跳过消费者自身作为汇聚点**（L135-L136）：若第二个消费者到达的节点本身是另一个消费者，说明结构退化，不构成"汇聚"。

**`tryMergeSource`（L156-L238）** 的守门：
1. 源 op 必须有 block_id、非 `arith.ConstantOp`、**单 tensor 结果**（标量没有轴语义）；(L159-L173)
2. 消费者条件：VECTOR、同 region、**block_id 与源块不同**、数量 ≥2；(L175-L191)
3. **汇聚点上游守卫（L200-L223）**：汇聚点的所有 operand 定义者，要么在待搬链里、要么在源块里——**不允许链外块的定义者**（否则搬走后汇聚点依赖悬空）。这是本 pass 最精细的正确性约束；
4. `willCreateCycle` 通过后，整条链 `updateBlockId` 到源块。(L228-L237)

### 3.6 MergeSmallBlockPass（MergeSmallBlockPass.cpp）

**做什么**：把 **≤3 个计算 op 的小 Vector 块**并入它的操作数块（上游）或用户块（下游）。块太小则块间通信开销占比过高，合并进邻居更划算。

**什么是"计算 op" `isTensorComputeOp`（L93-L122）**：
- **legacy 内核特判**（L94-L100）：`pcb10_tc01_kernel` 函数用旧的宽松判定 `isTensorComputeOpLegacy`（全 tensor 操作数 + 全 tensor 结果）；
- `linalg::LinalgOp`（L101-L115）：排除 copy、broadcast interface、以及用常量填充的 `linalg.fill`（这些不是纯计算）；
- `Elementwise` trait 且含 tensor 结果（L117-L121）。

**候选收集（L559-L678）**：

```cpp
collectUpstream(...)    // 操作数定义者所属块：必须是 VECTOR、有 block_id；
                        // 若定义者来自 CUBE 或控制流 → 无上游候选；
                        // 全部标量依赖 → 无上游候选；
                        // 恰 1 个上游块且无环 → 加入候选
collectDownstream(...)  // 结果用户所属块：同 block 用户忽略；
                        // 有 CUBE/控制流用户 → 无下游候选；
                        // 每个下游块无环 → 加入候选
```

关键细节：上游/下游候选都会被"**CUBE 链接**"短路（L564、L618）：小块的任何操作数定义者/结果用户如果是 CUBE 块（`haveCubeLink`），则该方向整体放弃——**Vector 小块不能并入 CUBE 块，反之亦然**（core 边界不可跨越）。

**目标仲裁 `selectMergeTarget`（L680-L705）**——策略链（顺序敏感）：

```cpp
strategies.push_back(std::make_unique<FaFWDPatternStrategy>());  // 1) 特化模式
strategies.push_back(std::make_unique<UBOccupationStrategy>());  // 2) UB 节省量
strategies.push_back(std::make_unique<IROrderStrategy>());       // 3) IR 程序顺序
strategies.push_back(std::make_unique<DefaultUpStrategy>());     // 4) 兜底向上
```

1. **`FaFWDPatternStrategy`（L371-L557）**：识别 FlashAttention 前向 softmax 三件套 `exp + mulf + addf`：
   - `exp(subf)`：subf 来自上游块（upBlockId），exp 结果同时被 mulf 和下游的 broadcast 用；(L392-L431)
   - `mulf(exp, BlockArgument)`：另一操作数是**循环迭代参数**，结果只被 addf 用；(L433-L456)
   - `addf(mulf, reduce)`：reduce 来自上游块，结果**唯一用户是 yield，且 yield 更新的正是 mulf 用的那个迭代参数**（`forOp.getRegionIterArg(yieldOpIndex) == mulfArg`）。(L458-L522)
   - 匹配成功 → **并入下游块**（downBlockIds[0]）。这是"渐进式 reduce 更新"的经典模式，必须跟着循环参数走。
2. **`UBOccupationStrategy`（L198-L320）**：计算并入每个候选块的 **UB 节省量**：
   - 上游块：`getOperandUB`（L231-L255）——小块操作数中定义者在候选块的（非 `tensor.empty`）数据体积之和（这些值并入后不再跨块）；
   - 下游块：`getUserUB`（L257-L278）——小块结果中**所有用户都在候选块**的体积之和（结果不再跨块）；用户跨多块则不算；
   - 取**最大节省量**的候选（`MAX_EDGE_SIZE` 视为无穷大不可取）。这是"向最省钱的方向并"。
3. **`IROrderStrategy`（L171-L196）**：按 `id2order`（块首次出现在 IR 中的顺序）选**最近的下游块**（若存在），否则全上游候选保留。保证合并方向符合程序顺序、块保持连续。
4. **`DefaultUpStrategy`（L322-L339）**：兜底，直接选第一个上游候选；无上游则第一个下游。

**两轮调度（L775-L780 + L832-L839）**：

```cpp
// 该 pass 在流水线里被调度两次。
bool firstRun = !module->getAttrOfType<BoolAttr>(kMergeSmallBlockFirstRunDone);
module->setAttr(kMergeSmallBlockFirstRunDone, BoolAttr::get(...));

// ... 选中目标后：
if (matchSIToFPSubMulPattern(ops)) {          // sitofp→subf→mulf 模式
  if (firstRun) {                             // 第一轮：推迟
    LOG_DEBUG("Defer sitofp-subf-mulf pattern merge ... to second run");
    continue;
  }
  markSubBlockOps(nowBlockId, targetBlockId, bm);  // 第二轮：标记 kSubBlock
}
```

**为什么推迟**：`sitofp→subf→mulf` 链（`matchSIToFPSubMulPattern` L727-L752）的第一轮先让其他块合并产生新的上下文，第二轮这个模式才有正确的合并目标。这也是 `firstRun` 标记（Module 属性）存在的意义——与 `kMergeComputeBlockApplied` 同类的"pass 间状态传递"。

**`markSubBlockOps`（L757-L766）**：第二轮处理该模式时，对源块和目标块的每个 op 打 `kSubBlockId`——记录"这些 op 原属哪块"，供下游（如传输/degrade pass）追溯合并历史。

### 3.7 RelocateMemrefDeclPass（RelocateMemrefDeclPass.cpp）

**做什么**：把**跨同步 op 的 memref 声明**连同整个依赖闭包搬到消费者块。理由（L294-L296 注释）：SplitDataflow 降同步栅栏时，若 memref 声明在栅栏一侧、使用在另一侧，会被拆坏；必须在降栅栏前让声明和使用同侧。

**Pass 入口（L272-L300）**：

```cpp
void RelocateMemrefDeclPass::runOnOperation() {
  if (CVPipeline::hasFallbackAttr(module)) { return; }
  // 没有任何同步 op → 无事可做（早退，省一次全图扫描）
  bool hasSync = false;
  module.walk([&](Operation *op) {
    if (CVPipeline::isSyncOp(op)) { hasSync = true; return WalkResult::interrupt(); }
    return WalkResult::advance();
  });
  if (!hasSync) { return; }
  relocateMemrefDecls(module);
}
```

**核心 `relocateMemrefDecls`（L164-L253）**，两步：

**① 扫描 + 闭包构建（L174-L236）**：

```cpp
module.walk([&](Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!isa<MemRefType>(operand.getType())) continue;      // 只看 memref 操作数
    Operation *producer = operand.getDefiningOp();
    if (!producer) continue;

    // 确定 producer 与 consumer 的公共 block，并检查中间是否有同步 op
    Block *commonBlock = producer->getBlock();
    ...
    CVPipeline::SyncWall &wall = walls.try_emplace(commonBlock, commonBlock).first->second;
    if (!wall.hasSyncBetween(pAnchor, cAnchor)) continue;   // 无同步 → 无需搬家

    Block *targetBlock = op->getBlock();
    int targetId = bm.getBlockIdByOp(op);
    if (targetId == -1) continue;                            // 消费者还没分块

    Operation *anchor = cAnchor;
    SmallVector<Operation *> ordered = buildForwardClosure(  // 构建前向闭包
        producer, targetBlock, anchor, pos, wall, domInfo);
    if (ordered.empty()) continue;
    if (commonBlock == targetBlock &&
        !wall.hasSyncBetween(producer, anchor)) continue;    // 最早使用必须跨过同步
    if (!allOperandsDominate(cluster, anchor, domInfo)) continue;  // 支配检查
    worklist.push_back({anchor, ordered, targetId, op->getAttr(kCoreType)});
  }
});
```

- **同步检测（L199-L202）**：`SyncWall`（按 block 懒创建）判 producer→consumer 之间是否有同步 op。
- **`buildForwardClosure`（L73-L145）**——本 pass 最核心的算法，**前向 + 后向不动点闭包**：
  - 前向（L84-L102）：对簇内每个 op 的用户——若用户已在目标块（`getAncestorInBlock`），更新 `anchor` 为**最早的跨同步用户**（`isBeforeInBlock` + `wall.hasSyncBetween`）；若该用户与当前 op 之间也有同步，则把用户也拉入簇；若用户是**terminator**（如循环携带的 yield）→ 整体失败（不能跨 terminator 下沉）。
  - 后向（L104-L132）：对簇内 op 的 memref 操作数定义者——若定义者**不支配目标块**且非簇成员，则拉入簇（前提：它的所有用户都随簇一起走，否则外移会破坏外部用户）；定义者是 terminator → 失败。
  - 循环直到不动点（`changed` 标志）。
  - 簇按原始 IR 顺序 `stable_sort`（用 `pos` 映射，L140-L144），保证搬家后顺序稳定。
- **`allOperandsDominate`（L48-L62）**：簇内每个成员的操作数，要么定义者在簇内（一起搬）、要么定义者已支配 anchor——保证搬到 anchor 前 SSA 支配成立。

**② 物理搬移（L238-L252）**：

```cpp
for (const Relocation &reloc : worklist) {
  auto insertPt = reloc.anchor;
  for (auto it = reloc.cluster.rbegin(); it != reloc.cluster.rend(); ++it) {
    auto member = *it;
    member->moveBefore(insertPt);        // 逆序逐个插到 anchor 前，保持原顺序
    insertPt = member;
    bm.updateBlockId(member, reloc.targetId);   // 同步改写 block_id 和 core_type
    ...
  }
}
```

**要点**：
- `moveBefore` 是**物理搬移**（区别于前几个 pass 只改 block_id）——因为跨同步的声明不搬走的话，物理顺序仍会被同步栅栏卡住；
- `bm.updateBlockId` 同步更新归属（L247），`coreType` 属性也已记录在 Relocation 里（L234，虽然此处只存未直接使用）；
- 只移动 **memref 声明**（alloc/声明类 op），不移动计算——计算移动由其他 pass（MoveLoadIntoUser 等）负责。

---

## 四、算法流程总结

```
02_UBAndLoadOpt（7 个子 Pass）
├─ 3.1 UnifyAllocBlock    alloc + scf.if(fill) + subview 链 → 统一到 copy 的块
│    直接用户(除fill) → 找if内fill → 公共block_id(穿透subview找copy)
│    → if无杂op → 收集闭包 → 环检 → updateBlockId
├─ 3.2 UBUsageOpt         加权依赖图上找最小 UB 切分点
│    权重: Tensor=体积 / Memref=MAX / Index=0；CUBE同块收缩
│    → 建SSA+内存边(自环剔除) → VECTOR候选(active end node)
│    → 单链上最小割点 → 组级环检 → updateBlockIdWithInner
├─ 3.3 BroadcastUBOpt     broadcast 全部用户同块 → 下沉到用户块
├─ 3.4 PosMaskPattern     broadcast+eq cmp+sle cmp+2×extui 整组下沉
├─ 3.5 MergeSameSourceAxis ≥2 消费者 BFS 找汇聚点 → 汇聚链并入源块
├─ 3.6 MergeSmallBlock    ≤3 计算 op 小块并入邻居（两轮调度）
│    候选: 上下游(排除CUBE链接) → 仲裁: FaFWD→UB节省→IR顺序→默认上
└─ 3.7 RelocateMemrefDecl 跨同步的 memref 声明 + 前向闭包 → 搬到消费者块
      buildForwardClosure(前向用户+后向定义者不动点) → 支配检 → moveBefore
```

**整体策略**：这 7 个 pass 的共性是把"数据结构与块边界错位"修复掉——声明跟着使用走（3.1/3.7）、小数据就近消费（3.3/3.4）、共享源聚拢（3.5）、碎片块并入邻居（3.6）、超 UB 的依赖切开（3.2）。其中 **UBUsageOpt 是唯一的"切分"pass**（把块分开），其余都是"合并/搬迁"pass（把块并拢或挪动）——先通过切分确认容量边界，再通过合并减少块数。

---

## 五、面试要点

1. **UBUsageOpt 为什么叫"UB usage"优化？它和块合并的关系是什么？**
   它把"两个块共享的 UB value 大小"作为图边权重，在**最小权重处切分**计算块——即"两块共用 UB 且生命周期重叠 → 保持/切分离，避免超容量"。它是 02 专题里**唯一做"分"的 pass**；其余 6 个都做"合/搬"。顺序上先分后合：先确定哪些必须分开（容量约束），再合并小块（减少块数）。

2. **为什么 Memref 的边权重是 MAX_EDGE_SIZE？**
   memref 只是内存声明，本身不占 UB。设成最大值是为了**禁止在图建模时把 memref 跨块切分**——否则声明和它的 load/store 会分属两块，后续降栅栏/重排会破坏"声明支配使用"的不变量。这是"用权重表达非法切分"的建模技巧。

3. **CUBE 块在图里为什么被收缩成一个节点？**
   UB 优化只关心 Vector 侧的缓冲占用。同 block_id 的 CUBE op 内部依赖不影响 Vector 的 UB 切分，收缩成单节点后：① 图规模大幅减小；② 所有指向该 CUBE 块的边自然汇聚到同一个节点，切分决策只发生在 CUBE↔Vector 边界。**建模粒度 = 决策粒度**。

4. **`getCommonBlockId` 为什么对 subview 要"穿透到 memref.copy"？**
   subview 是纯视图 op，没有独立语义归属；真正决定 alloc 归属的是读它的 `memref.copy`。若直接取 subview 的 block_id，会拿到一个"视图层"的中间归属，可能与 copy 所在块不一致，导致 alloc 被划到错误的一侧。穿透后取**数据消费点**的块，语义正确。若有多个 copy 在不同块则报错（failure → fallback）——无法唯一确定归属时宁可不优化。

5. **UnifyAllocBlock 中 scf.if 必须满足哪些条件才允许统一？为什么？**
   ① fill 在 if 的 then region 且 if 无 else（三明治结构）；② then region 里除 fill 外无其他 op。理由：整个 `scf.if` 会被划进目标块，如果 if 里混了别的逻辑，统一 block_id 会把无关计算也带走，破坏其他块的依赖。**安全第一，模式不匹配就跳过**。

6. **MergeSmallBlock 的目标仲裁为什么用"策略链"且顺序敏感？**
   小块并入哪个方向有多目标权衡：FlashAttention 的 softmax 更新模式必须跟着循环参数（FaFWD，语义驱动）；否则看 UB 节省量（UBOccupation，性能驱动）；再否则看 IR 顺序（IROrder，保持程序局部性）；最后默认向上游。**先语义、后性能、再启发式、最后兜底**——每个策略都是"更优方向"的启发式，一个收敛就停止。

7. **MergeSmallBlock 为什么要跑两轮？**
   该 pass 在流水线里调度两次（L76 和 L76 后的重排再调度）。`sitofp→subf→mulf` 模式第一轮先让普通小块合并，产生新的块上下文；第二轮该模式才有正确的合并目标（并入后与周围块的依赖才干净）。用 Module 属性 `kMergeSmallBlockFirstRunDone` 区分两轮——与 `kMergeComputeBlockApplied` 同类机制。

8. **RelocateMemrefDecl 与 UnifyAllocBlock 有什么异同？**
   相同点：都处理"memref 声明归属与使用方错位"。不同点：① 触发条件——UnifyAllocBlock 看"消费模式"（copy 在哪个块），RelocateMemrefDecl 看"是否跨同步 op"；② 手段——前者只改 block_id，后者 `moveBefore` 物理搬移 + 改 block_id；③ 范围——后者用 `buildForwardClosure` 把整个**前向/后向依赖闭包**一起搬，因为跨同步的声明必须"整簇"移动才能保持支配。RelocateMemrefDecl 是**降同步栅栏前的最后一个准备**。

9. **`buildForwardClosure` 为什么是"前向+后向不动点"？**
   只搬 producer 本身不够：producer 的结果可能被目标块外的 op 使用（前向），producer 的 memref 定义者可能不在目标块（后向）。必须把"随 producer 一起搬的所有使用者和定义者"递归收齐到不动点，且每个成员的每个操作数都满足支配（`allOperandsDominate`），搬家才不会破坏 SSA。**闭包的不动点迭代 + 支配验证 = 安全的图搬移**。

10. **这些 pass 为什么都围绕 `ssbuffer.block_id` 而不是直接改 IR 结构？**
    CV 流水线的核心抽象是"计算块"。所有后续 pass（Reorder、SplitDataflow、传输插入、多缓冲分配）都基于 block_id 做调度。优化阶段只需要把 block_id 这个**决策变量**改对，物理重排交给 ReorderOpsByBlockId 统一完成——**关注点分离**：Opt 只改归属，Reorder 只改顺序。
