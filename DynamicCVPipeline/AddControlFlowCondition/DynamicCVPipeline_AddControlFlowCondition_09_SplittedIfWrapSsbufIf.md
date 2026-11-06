# DynamicCVPipeline 子 Pass 专题：splitted_if 包装 ssbuffer.if（09）

> 覆盖 `SplittedIfWrapSsbufIfPass`（[SplittedIfWrapSsbufIf.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/SplittedIfWrapSsbufIf.cpp)）
>
> 流水线位置：`AddControlFlowCondition` 编排的第 7 步（也是最后一步），`UpdateLoopIterTimes`（Step 6）之后、`SeparateMemoryFromCompute`（L84）之前。
>
> 职责：把 `SplitIfByBlockId` 拆出的 **splitted_if** 与 `AllocMultiCache` 注入的 **ssbuffer.if** 这两个独立 if 合并成**一个**外层迭代分派 if——用 splitted_if 的条件新建外层 if 把 ssbuffer.if 包进 then 分支，再展开原 splitted_if，使"分裂后的块"与"多缓冲轮询"共享同一循环迭代条件。

---

## 一、这个专题在做什么

前面几步完成后，同一个主循环体内同时存在两种 if：

1. **splitted_if**（带 `kSplittedIf` 标记）：`SplitIfByBlockId` 依据 block_id 把一个大 if 分裂成多个小块 if。每个 splitted_if 的`条件`决定"本次迭代走这块核（CUBE 组 / Vector 组）还是跳过"。
2. **ssbuffer.if**（带 `kIf` 标记）：`AllocMultiCache` 为每个核间/核内传输依赖注入的多缓冲轮询 if，条件决定"本迭代用原缓冲还是输出缓冲"。

这两个 if 逻辑上描述的是**同一个块的两次分派**，却被拆成了两个独立 if。它们在 then/else 分支里其实各持有一部分该块的运算：splitted_if 的 then 里有真正的张量计算，ssbuffer.if 的 then 里有缓冲轮询的 wait/transfer/set。二者嵌套关系在 IR 上不固定（可能是 splitted_if 包 ssbuffer.if，也可能相反），导致：

- 同一个 block 的运算被两个条件叠加控制，语义冗余；
- `RemoveSsbufAttr` / 后续 codegen 需要处理的 if 层级更复杂；
- 某些块里 splitted_if 的条件 def-chain 落在 ssbuffer.if 的 then 内，跨 if 使用 SSA 值，破坏区域局部性。

`SplittedIfWrapSsbufIf` 的做法是**统一收口**：

- **提条件**：把 splitted_if 的条件（及其在 ssbuffer.if then 内的 def-chain）hoist 到 ssbuffer.if 之上；
- **包外层**：用 splitted_if 的条件新建一个 `newIfOp`（继承 `kSplittedIf` 标记），把 ssbuffer.if 整个挪进它的 then；
- **重造 else**：在 newIfOp 的 else 里重建 ssbuffer.if 的 else 语义（for 里计数器 +step，while 里克隆 while_arg 并重映射）；
- **展开旧 splitted_if**：把原 splitted_if 的 then 体展开到 newIfOp 外的当前位置，删除 splitted_if，再把新 if 的结果回填给原 ssbuffer.if 的外部 uses。

最终每个块只剩**一个** if（newIfOp，条件 = 分裂分派条件），内部嵌套原 ssbuffer.if（条件 = 缓冲轮询条件）。语义等价于"该块在本轮被分裂选中时，再进行缓冲轮换"。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kSplittedIf`（`ssbuffer.splitted_if`） | `SplitIfByBlockId` 给分裂块打的唯一 int 标记，`splittedIfId` 相同表示同一次分裂的多个块 |
| `kIf`（`ssbuffer.if`） | `CreateIfOps` 给每个 block_id 生成的 if 标记，`ssbuffer.if` 的块身份 key |
| `kWhileArg`（`ssbuffer.while_arg`） | while 循环中由 per-block arg 克隆而来的 op 标记（`ProcessArgs` 产物），`buildElseUpdate` 的 while 分支据此识别需在 else 重建的 op |
| `ssbuffer.if` / `ssbufIf` | 多缓冲轮询 if（`AllocMultiCache` 注入），本 Pass 的"被包装对象" |
| `splitted_if` / `splittedIf` | 分裂块 if（`SplitIfByBlockId` 注入），本 Pass 的"条件来源"，包装后被展开删除 |
| `newIfOp` | `wrapSsbufIf` 新建的外层 if（用 splitted_if 条件，继承 kSplittedIf），最终取代 splitted_if |
| `MainLoop::getBody()` | for 返回 body；while 返回 **after body**（本 Pass 只处理直属于主循环 body 的 `ssbuffer.if`，嵌套 if 不属于该 block 的 op 列表） |
| `ControlFlowConditionInfo::cntArgs` | `ifOp -> 该 if 使用的 loop iter_arg 计数器`（`UpdateLoopOps` 注入），`buildElseUpdate` 的 for 分支用它对计数器作 `+step` |
| `isRealTensorComputeOp` | 真正的张量算术/规约 op（`isTensorComputeOp` 剔除 copy/broadcast/fill；再剔除 `linalg.transpose` 与 `arith.select`）——用来判定"if 内是否有真正计算" |
| `then/else passthrough` | ssbuffer.if 的 else yield 操作数：else 分支"默认不换缓冲"的透传值，重建 newIfOp else 时作为初值 |
| `stayOutside`（`moveDefChainBefore` 参数） | 不允许移动的 op 集合：splitted_if 的条件 def-chain，保持原位供 splitted_if 使用 |

---

## 三、逐行讲解

### 3.1 入口编排：runOnOperation（L525-L552）

```cpp
void SplittedIfWrapSsbufIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;   // 上游已回退 → 短路

  // Main loop -> (ssbuffer.if, its one splitted_if)
  llvm::MapVector<Operation *, SmallVector<std::pair<scf::IfOp, scf::IfOp>>>
      loopToSsbufSplittedIf;
  if (!checkCanWrap(module, loopToSsbufSplittedIf)) return;   // 见 3.10

  for (auto &[loopOp, ssbufIfs] : loopToSsbufSplittedIf) {
    for (auto [ssbufIf, splittedIf] : ssbufIfs) {
      if (splittedIfWrapSsbufIf(ssbufIf, splittedIf, loopOp, info) ==
          WRAP_SSBUF_IF_SUCCESS)
        continue;
      LDBG("splittedIfWrapSsbufIf failed. if: " << ssbufIf << "\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }
}
```

**要点**：
- `loopToSsbufSplittedIf`（`MapVector`）保存"每主循环里每个需包装的 `(ssbuffer.if, 其唯一 splitted_if)` 对"；
- **任一包装失败 → `ERRCODE_FAILED` 整体 fallback**（不静默跳过，因为包装到一半的 if 结构已被改动，继续执行会让 IR 处于半包装状态）；
- 包装对之间的独立性强，但 IR 改动会互相影响位置，故逐对顺序处理，失败即停。

### 3.2 主包装流程：splittedIfWrapSsbufIf（L374-L407）

```cpp
static int splittedIfWrapSsbufIf(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                                 Operation *loopOp,
                                 ControlFlowConditionInfo *info) {
  // ① 快照 ssbuffer.if 的 else yield（透传值，作 newIfOp else 的初值）
  auto elseYield = cast<scf::YieldOp>(ssbufIf.elseBlock()->getTerminator());
  SmallVector<Value> elsePassthrough(elseYield.getOperands().begin(),
                                     elseYield.getOperands().end());

  // ② 提 splitted_if 条件 + 打包 ssbuffer.if 条件（见 3.8）
  hoistSplitCondAndPackSsbufCond(ssbufIf, splittedIf);

  // ③ 新建外层 if，把 ssbuffer.if 挪进 then（见 3.6）
  scf::IfOp newIfOp;
  wrapSsbufIf(ssbufIf, splittedIf, newIfOp);

  // ④ 重建 newIfOp 的 else（for：计数器 +step；while：克隆 while_arg）
  Location loc = ssbufIf.getLoc();
  if (buildNewIfElse(loopOp, info, ssbufIf, newIfOp.elseBlock(),
                     elsePassthrough, loc) != WRAP_SSBUF_IF_SUCCESS)
    return WRAP_SSBUF_IF_FAILED;

  // ⑤ 展开原 splitted_if（其 then 体挪到当前处，删 if）
  unwrapSplittedIf(splittedIf);

  // ⑥ newIfOp then 的 yield 改 yield ssbuffer.if 的结果
  Block *thenBlock = newIfOp.thenBlock();
  if (thenBlock->mightHaveTerminator()) thenBlock->getTerminator()->erase();
  OpBuilder thenBuilder(thenBlock, thenBlock->end());
  thenBuilder.create<scf::YieldOp>(loc, ssbufIf.getResults());

  // ⑦ 把 newIfOp 外的 ssbuffer.if 结果 uses 重定向到 newIfOp 的结果
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), newIfOp.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !newIfOp->isProperAncestor(use.getOwner());
    });
  }
  return WRAP_SSBUF_IF_SUCCESS;
}
```

**要点**：整体是一段"外科手术式"的重构，五个动作环环相扣——先快照 else 透传值（这是 newIfOp else 的语义基线），再提条件、包外层、重造 else，最后展开 splitted_if 并**重定向 uses**。关键细节（⑦）：`replaceUsesWithIf` 只替换 `newIfOp` 外部的 uses——`newIfOp` 内部（then 里的 ssbuffer.if）仍用原结果，通过 then yield 传给 newIfOp 的结果。

### 3.3 展开 splitted_if：unwrapSplittedIf（L60-L78）

```cpp
static void unwrapSplittedIf(scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
  SmallVector<Value> thenVals(thenYield.getOperands().begin(),
                              thenYield.getOperands().end());

  // 收集 then 内所有非终结 op，逐个移到 splitted_if 之前
  SmallVector<Operation *> thenOps;
  for (Operation &op : *thenBlock)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      thenOps.push_back(&op);
  for (Operation *op : thenOps) op->moveBefore(splittedIf);

  // 用 then yield 的操作数替换 splitted_if 的所有 uses，再删 if
  splittedIf.replaceAllUsesWith(thenVals);
  splittedIf.erase();
}
```

**要点**：这是标准"纯 then if 展开"——把 then 体提到 if 前、uses 替换成 then yield 值、删 if。前提是 `computeOnlyInSplitThen` 已保证"真正的张量计算都在 splitted_if 的 then 里"，所以展开后 then 体的计算不会丢失，else 分支不含真实计算（见 3.10 的门禁）。

### 3.4 else 更新信息构建：buildElseUpdate（L83-L120）

```cpp
static int buildElseUpdate(Operation *loopOp, ControlFlowConditionInfo *info,
                           scf::IfOp ssbufIf, OpBuilder &elseBuilder,
                           ArrayRef<Value> elsePassthrough, Location loc,
                           IRMapping &mapping) {
  // for：把 block 计数器映射为 counter + step
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    if (!info->cntArgs.count(ssbufIf)) return WRAP_SSBUF_IF_FAILED;   // 无计数器
    Value counter = info->cntArgs[ssbufIf];
    if (!counter) return WRAP_SSBUF_IF_FAILED;
    Value inc = elseBuilder.create<arith::AddIOp>(loc, counter, forOp.getStep());
    mapping.map(counter, inc);
    return WRAP_SSBUF_IF_SUCCESS;
  }

  // while：克隆所有带 kWhileArg 的 op，并把 else 操作数映射到 then 值克隆
  auto thenYield = cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  for (Operation &op : *ssbufIf.thenBlock()) {
    if (!op.hasAttr(kWhileArg)) continue;
    Operation *cloned = elseBuilder.clone(op, mapping);
    elseBuilder.setInsertionPointAfter(cloned);
  }
  for (auto [idx, thenVal] : llvm::enumerate(thenYield.getOperands())) {
    if (!mapping.contains(thenVal)) continue;
    mapping.map(elsePassthrough[idx], mapping.lookup(thenVal));
  }
  return WRAP_SSBUF_IF_SUCCESS;
}
```

**要点**：newIfOp 的 else 要重现 ssbuffer.if 的 else 语义——"走 else 分支的那一半缓冲的变量更新"：

- **for**：else 里只需把块计数器 `+step`（因为迭代由归纳变量驱动，else 侧也要让计数器随迭代前进，否则下一轮的轮询奇偶判定错位）；
- **while**：while 无归纳变量，计数器靠 `ssbuffer.while_arg` 标记的 op 链显式维护，所以要把这些 op **克隆进 else**，并把 `elsePassthrough[idx]`（else 的透传操作数）重映射到 then 对应值的克隆。

### 3.5 重建 else 块：buildNewIfElse（L124-L157）

```cpp
static int buildNewIfElse(Operation *loopOp, ControlFlowConditionInfo *info,
                          scf::IfOp ssbufIf, Block *elseBlock,
                          ArrayRef<Value> elsePassthrough, Location loc) {
  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator())
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());

  IRMapping mapping;
  if (buildElseUpdate(loopOp, info, ssbufIf, elseBuilder, elsePassthrough, loc,
                      mapping) != WRAP_SSBUF_IF_SUCCESS)
    return WRAP_SSBUF_IF_FAILED;

  // elseVals 初值 = 透传值，被 mapping 命中的槽位换成刚构建的新值
  SmallVector<Value> elseVals(elsePassthrough.begin(), elsePassthrough.end());
  for (Value &val : elseVals)
    if (mapping.contains(val))
      val = mapping.lookup(val);

  if (elseVals.size() != ssbufIf.getNumResults())
    return WRAP_SSBUF_IF_FAILED;   // 结果数不匹配 → 失败

  if (elseBlock->mightHaveTerminator()) elseBlock->getTerminator()->erase();
  OpBuilder yieldBuilder(elseBlock, elseBlock->end());
  yieldBuilder.create<scf::YieldOp>(loc, elseVals);
  return WRAP_SSBUF_IF_SUCCESS;
}
```

**要点**：`newIfOp.elseBlock()` 是 `create<scf::IfOp>(..., withElse=true)` 里预置的空 else（自带一个空 yield）。这里按 `elsePassthrough`（原 ssbuffer.if 的 else yield 操作数）重建，`buildElseUpdate` 通过 `mapping` 指出哪些槽位要被更新（for 的计数器 / while 的 while_arg 克隆）。**结果数校验**是保护：newIfOp 的结果类型 = ssbuffer.if 的结果类型，else yield 操作数必须一一对应。

### 3.6 新建外层 if 并打包：wrapSsbufIf（L296-L320）

```cpp
static void wrapSsbufIf(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                        scf::IfOp &newIfOp) {
  Location loc = ssbufIf.getLoc();
  OpBuilder builder(ssbufIf);
  SmallVector<Type> resultTypes(ssbufIf->getResultTypes());
  // 用 splitted_if 的条件新建外层 if，结果类型同 ssbuffer.if
  newIfOp = builder.create<scf::IfOp>(loc, resultTypes, splittedIf.getCondition(),
                                      /*withElse=*/true);
  // 继承 splitted_if 的 kSplittedIf 标记（id 保持不变）
  newIfOp->setAttr(kSplittedIf, splittedIf->getAttr(kSplittedIf));

  // 把 ssbuffer.if 挪进 newIfOp 的 then（yield 前 / 末尾）
  Block *thenBlock = newIfOp.thenBlock();
  if (thenBlock->mightHaveTerminator()) ssbufIf->moveBefore(thenBlock->getTerminator());
  else ssbufIf->moveBefore(thenBlock, thenBlock->end());

  // 把 ssbuffer.if 的条件 def-chain 打包到 ssbuffer.if 之前
  DenseSet<Operation *> splitCondOps;
  collectConditionDefOpsBefore(splittedIf.getCondition(), newIfOp.getOperation(),
                               splitCondOps);
  moveDefChainBefore(ssbufIf.getCondition(), newIfOp.getOperation(),
                     ssbufIf.getOperation(), splitCondOps);
}
```

**要点**：`newIfOp` 的条件 = `splittedIf.getCondition()`（分裂分派条件），结果类型 = ssbuffer.if 的结果类型。`newIfOp` 继承 `kSplittedIf` 标记（同一个分裂 id），这样下游 `RemoveSsbufAttr`/其他 pass 仍能识别它是"分裂块"。末尾把 ssbuffer.if 的条件 def-chain 提到 ssbuffer.if 前，保证 then 内 ssbuffer.if 的条件不跨 if 引用外部。

### 3.7 条件 def-chain 搬运：moveDefChainBefore（L236-L292）

这是本 Pass 最核心的 SSA 维护工具，配套 4 个小函数：

```cpp
// ① 收集 anchor 所在 block 内、位于 anchor 之前、且属于 value 的 def-chain 的 op
static void collectConditionDefOpsBefore(Value value, Operation *anchor,
                                         DenseSet<Operation *> &ops) {
  SmallVector<Value> worklist;
  if (value) worklist.push_back(value);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    Operation *defOp = current.getDefiningOp();
    if (!defOp || defOp->getBlock() != anchor->getBlock() ||
        !defOp->isBeforeInBlock(anchor) || !ops.insert(defOp).second)
      continue;
    worklist.append(defOp->operand_begin(), defOp->operand_end());
  }
}

// ② 收集 anchor 之前、operand 由 ops 里 op 定义的 annotation.mark（"卫星"标记 op）
static void addSatelliteMarkOps(DenseSet<Operation *> &ops, Operation *anchor);

// ③ op 的所有 uses 是否都"局部"（在 ops 里 / insertBefore 内部 / 同 block 更前）
static bool opHasOnlyLocalUses(Operation *op, const DenseSet<Operation *> &ops,
                               Operation *insertBefore);

// ④ 找到 condOps 里已位于 before 之前的第一个 op，否则返回 before
static Operation *insertPointForCondOps(Operation *before,
                                        const DenseSet<Operation *> &condOps);
```

`moveDefChainBefore` 主体（L245-L291）：

```cpp
static void moveDefChainBefore(Value cond, Operation *fromAnchor,
                               Operation *before,
                               const DenseSet<Operation *> &stayOutside) {
  DenseSet<Operation *> condOps;
  collectConditionDefOpsBefore(cond, fromAnchor, condOps);
  if (condOps.empty()) return;
  addSatelliteMarkOps(condOps, fromAnchor);

  // 按 block 顺序取出 condOps 里位于 fromAnchor 前的 op
  SmallVector<Operation *> sorted;
  for (Operation &op : *fromAnchor->getBlock()) {
    if (condOps.contains(&op)) sorted.push_back(&op);
    if (&op == fromAnchor) break;
  }

  // moveSet：可移动 = 不在 stayOutside 且 uses 全局部（第一次筛选）
  DenseSet<Operation *> moveSet;
  for (Operation *op : sorted)
    if (!stayOutside.contains(op) && opHasOnlyLocalUses(op, condOps, before))
      moveSet.insert(op);

  // 反着扫：若某 op 有"留在原地"的 user，则它自己也得留下（反向传播）
  for (Operation *op : llvm::reverse(sorted)) {
    if (!moveSet.contains(op)) continue;
    for (OpOperand &use : op->getUses()) {
      Operation *user = use.getOwner();
      if (condOps.contains(user) && !moveSet.contains(user)) {
        moveSet.erase(op);
        break;
      }
    }
  }

  // 跨 block 时，插入点取 condOps 里已位于 before 之前的最前 op
  Operation *insertPt = before;
  if (fromAnchor->getBlock() != before->getBlock())
    insertPt = insertPointForCondOps(before, condOps);

  for (Operation *op : sorted)
    if (moveSet.contains(op) && op != insertPt)
      op->moveBefore(insertPt);
}
```

**要点**：
- **`collectConditionDefOpsBefore`** 是做"值的 def-chain 切片"：从 `cond` 出发，沿 def 向上把"同 block 且在该 anchor 之前"的 op 全部收进 `condOps`（一个封闭的依赖闭包）；
- **`addSatelliteMarkOps`** 处理 `annotation.mark`：标记 op 的 operand 若由 condOps 定义，则标记本身也要一起搬（否则标记引用悬空/错位）；
- **两次筛选**保证只搬"可整体独立搬走"的子图：先按"全局部 uses"挑候选，再反向传播——如果一个候选 op 的 user 里有 `condOps` 成员但那个 user 不搬（因为它有外部 user），那么这个 op 也得留下；
- `stayOutside` 是 splitted_if 的条件 def-chain，需要保持原位供 splitted_if 继续使用，不能搬进 then。

### 3.8 提条件 + 打包条件：hoistSplitCondAndPackSsbufCond（L342-L370）

```cpp
static void collectConditionDefOpsInside(Value value, scf::IfOp ssbufIf,
                                         DenseSet<Operation *> &ops) {
  // 与 collectConditionDefOpsBefore 对称，但限定 def 在 ssbufIf 内部（isProperAncestor）
  ...
}

static void hoistSplittedIfCondition(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  DenseSet<Operation *> condOps;
  collectConditionDefOpsInside(splittedIf.getCondition(), ssbufIf, condOps);
  if (condOps.empty()) return;
  SmallVector<Operation *> sorted;
  for (Operation &op : *ssbufIf.thenBlock())
    if (condOps.contains(&op)) sorted.push_back(&op);
  for (Operation *op : sorted) op->moveBefore(ssbufIf);   // 提到 ssbufIf 前
}

static void hoistSplitCondAndPackSsbufCond(scf::IfOp ssbufIf,
                                           scf::IfOp splittedIf) {
  hoistSplittedIfCondition(ssbufIf, splittedIf);          // ①
  DenseSet<Operation *> splitCondOps;
  collectConditionDefOpsBefore(splittedIf.getCondition(), ssbufIf.getOperation(),
                               splitCondOps);
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  moveDefChainBefore(ssbufIf.getCondition(), ssbufIf.getOperation(),
                     ssbufIf.getOperation(), splitCondOps); // ②
}
```

**要点**：
- **① hoist 掉 splitted_if 的条件依赖**：若 splitted_if 的条件 def-chain 落进了 ssbuffer.if 的 then 里（跨 if 引用），先把这段 def-chain 提到 ssbuffer.if 之前；
- **② 打包 ssbuffer.if 的条件**：把 ssbuffer.if 自身的条件 def-chain 收到 ssbuffer.if 之前，同时 `stayOutside = splitCondOps`（splitted_if 的条件 def-chain 留在原位不动）。

### 3.9 实际张量计算判定：ssbufIfHasTensorCompute / computeOnlyInSplitThen / countSplittedIfs（L410-L459）

```cpp
// ssbuffer.if 内是否含"真张量计算 op"
static bool ssbufIfHasTensorCompute(scf::IfOp ssbufIf) {
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (isRealTensorComputeOp(op)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walk.wasInterrupted();
}

// 所有真张量计算是否都落在 splitted_if 的 then 里
static bool computeOnlyInSplitThen(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (thenBlock->findAncestorOpInBlock(*op)) return WalkResult::skip();
    if (isRealTensorComputeOp(op)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

// 统计 ssbuffer.if 下有几个 splitted_if（超过 1 个就中断）
static unsigned countSplittedIfs(scf::IfOp ssbufIf, scf::IfOp &splittedIf);
```

**要点**：这是 `checkCanWrap` 的安全门禁。只有"真正的张量算术/规约计算全部在 splitted_if 的 then 里"时，展开 splitted_if 才不会丢计算（`unwrapSplittedIf` 只搬 then 体）。若 ssbuffer.if 带有真实计算但**没有**对应 splitted_if → 无法包装，跳过。

### 3.10 可包装性检查：checkCanWrap（L463-L521）

```cpp
static bool checkCanWrap(ModuleOp module,
  llvm::MapVector<Operation *, SmallVector<std::pair<scf::IfOp, scf::IfOp>>>
      &loopToSsbufSplittedIf) {
  IntegerAttr splittedIfId;
  WalkResult loopWalk = module.walk([&](Operation *loopOp) -> WalkResult {
    if (!isMainLoopOp(loopOp)) return WalkResult::advance();

    // 只遍历主循环 body 里直属的 if（for body / while after body）
    Block *body = MainLoop(loopOp).getBody();
    SmallVector<std::pair<scf::IfOp, scf::IfOp>> ssbufIfs;
    for (Operation &op : body->without_terminator()) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp || !ifOp->hasAttr(kIf)) continue;
      scf::IfOp splittedIf;
      unsigned splittedIfCount = countSplittedIfs(ifOp, splittedIf);

      // >1 个 splitted_if：无法唯一确定 → 跳过整个循环
      if (splittedIfCount > 1) return WalkResult::interrupt();
      if (splittedIfCount == 1) {
        auto id = splittedIf->getAttrOfType<IntegerAttr>(kSplittedIf);
        if (!splittedIfId) splittedIfId = id;                 // 记录全局唯一 id
        else if (splittedIfId.getInt() != id.getInt()) return WalkResult::interrupt();
        if (!computeOnlyInSplitThen(ifOp, splittedIf)) return WalkResult::interrupt();
        ssbufIfs.push_back({ifOp, splittedIf});
        continue;
      }
      // 无 splitted_if 但含真计算：结构不支持 → 跳过
      if (ssbufIfHasTensorCompute(ifOp)) return WalkResult::interrupt();
    }
    loopToSsbufSplittedIf.insert({loopOp, std::move(ssbufIfs)});
    return WalkResult::advance();
  });
  if (loopWalk.wasInterrupted()) return false;
  if (!splittedIfId) return false;   // 全模块没有 splitted_if → 无事可做
  return true;
}
```

**要点**：
- **只处理主循环 body 直属的 `kIf`**（嵌套 if 不属于 `body->without_terminator()` 的 op 列表），符合"ssbuffer.if 直接位于主循环体"的结构预期；
- **三个中断条件**：某个 ssbuffer.if 含 >1 个 splitted_if、全模块出现多个不同 splitted_if id、或无 splitted_if 的 ssbuffer.if 里含真计算——任一命中即 `checkCanWrap` 返回 false，`runOnOperation` 直接 return（保守跳过，不 fallback，因为此时 IR 尚未改动，跳过是"能力不足但正确"）；
- **全局唯一 splitted_if id**：所有要包装的 splitted_if 必须共享同一个 id，否则无法确定"这个分裂分派条件属于哪个分裂批次"。

---

## 四、流程总结

### 4.1 SplittedIfWrapSsbufIf 的流水

```
SplittedIfWrapSsbufIfPass::runOnOperation
  ├─► hasFallbackAttr ? 短路返回
  ├─► checkCanWrap
  │     ├─► 遍历主循环（isMainLoopOp），body = MainLoop(loopOp).getBody()
  │     ├─► 直属 kIf -> countSplittedIfs 统计内部 splitted_if
  │     ├─► =0 且含真计算 -> interrupt；>1 -> interrupt；!=全局唯一 id -> interrupt
  │     └─► ==1 且 computeOnlyInSplitThen -> 记 (ssbufIf, splittedIf) 对
  └─► 逐对 splittedIfWrapSsbufIf
        ├─► 快照 ssbuffer.if else yield -> elsePassthrough
        ├─► hoistSplitCondAndPackSsbufCond   提 splitted_if 条件 + 打包 ssbuf 条件
        ├─► wrapSsbufIf                      新建 newIfOp（用 splitted_if 条件）包 ssbuf
        ├─► buildNewIfElse                   重建 newIfOp else（for +step / while 克隆）
        ├─► unwrapSplittedIf                 展开 splitted_if then，删除 splitted_if
        └─► 重定向 ssbuf 结果外部 uses -> newIfOp 结果
```

### 4.2 与前后子 Pass 的协作

- **前置**：`SplitIfByBlockId`（ComputeBlockOpt）打的 `kSplittedIf` 分裂标记；`CreateIfOps` 打的 `kIf`；`AllocMultiCache` 注入的 `ssbuffer.if` 多缓冲轮询；`UpdateLoopOps` 注入的 `cntArgs` 计数器；`ProcessArgs` 打的 `kWhileArg`；
- **产出**：每个块的"分裂分派 if"与"多缓冲轮询 if"合并为一个 `newIfOp`（外层分裂条件 + 内层缓冲轮询），并打上继承的 `kSplittedIf`；
- **后置**：`SeparateMemoryFromCompute`（L84）分离访存计算；`RemoveSsbufAttr`（L85）清理 `kSplittedIf`/`kIf` 等所有 `ssbuffer.*` 属性。

---

## 五、面试要点

1. **为什么需要 SplittedIfWrapSsbufIf？** 同一块的两层分派被拆成了两个独立 if：`SplitIfByBlockId` 的 `kSplittedIf`（按 block_id 分裂）与 `AllocMultiCache` 的 `kIf`（多缓冲轮询）。本 Step 用 splitted_if 的条件新建外层 if 包住 ssbuffer.if，再展开 splitted_if，把它们归一成一个迭代分派 if + 内部轮询，避免冗余条件叠加、简化控制流层级。

2. **为什么"包装"而非"内联"？** 两种 if 语义层级不同：split 条件决定"本块这一轮是否激活"，轮询条件决定"激活后用哪一半缓冲"。前者是外层 gate，后者是内层 select。用 splitted_if 条件做外层、ssbuffer.if 做内层，才符合"先分派到块、再在块内选缓冲"的执行语义。

3. **重复造 else 的难点在哪？** newIfOp 的 else 必须精确重现 ssbuffer.if 的 else 语义。for 循环只需把计数器 `+step`（`info->cntArgs`），while 循环没有归纳变量，必须把 `ssbuffer.while_arg` 标记的 op 克隆进 else 并把透传操作数重映射到 then 值克隆——否则 else 侧的变量更新丢失，轮询奇偶判定错位。

4. **`moveDefChainBefore` 两次筛选解决了什么？** 搬条件 def-chain 时不能破坏 SSA。第一次按"全局部 uses"挑候选，第二次反向传播"有留下 user 的 op 也留下"，保证只搬"能整体独立搬走"的闭合子图；`addSatelliteMarkOps` 避免 `annotation.mark` 引用悬空；`stayOutside` 保护 splitted_if 的条件 def-chain 不被误搬。

5. **`checkCanWrap` 为什么"跳过"而非"fallback"？** 三个中断条件（多 splitted_if、多 id、无 splitted_if 但含真计算）都属于"当前结构无法安全包装"，此时 IR 尚未改动，跳过只是不做优化、语义仍正确（等价于不合并两个 if）。而真正在 `splittedIfWrapSsbufIf` 中改到一半失败（如 else 结果数不匹配）才 `ERRCODE_FAILED`，因为 IR 已处于半包装状态。

6. **`isRealTensorComputeOp` 为什么剔除 transpose/select？** `isTensorComputeOp` 已剔除 copy/broadcast/fill；`linalg.transpose` 只是数据重排（无算术）、`arith.select` 是 `kSplittedIf` 周围的控制流胶水，都不构成"必须在 splitted_if then 里保存的真实计算"。若把它们算作计算，会误判"ssbuffer.if 含真计算"而错误跳过包装。

7. **展开 splitted_if 前为什么要求 `computeOnlyInSplitThen`？** `unwrapSplittedIf` 只搬 then 体、丢弃 else。若 then 之外还有真计算，展开后这些计算会丢失或错位。`computeOnlyInSplitThen` 保证"所有真张量计算都在 splitted_if 的 then 里"，展开才无损。