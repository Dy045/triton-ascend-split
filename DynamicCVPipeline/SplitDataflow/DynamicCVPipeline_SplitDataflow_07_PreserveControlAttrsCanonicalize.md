# SplitDataflow 子 Pass 逐行讲解（七）：PreserveControlAttrsCanonicalize

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.cpp`
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 6 步**（`SeparateCVScope` 之后）

---

## 一、这个 Pass 在做什么

在 `SeparateCVScope` 之后，IR 里多了很多"中性化"操作（0 常量、空 alloc 等），产生了大量可被 canonicalize 消除的冗余。这个 Pass 做标准的 MLIR canonicalize，但有一个关键前提：**在 canonicalize 过程中，必须保留控制流 op（`scf.for/if/while/parallel`）上先前打下的关键属性**（如 `ssbuffer.block_id`、`ssbuffer.main_loop` 等）。

如果直接做普通 canonicalize，MLIR 可能会把某个控制流 op 替换/消除成新的 op，导致这些标记属性**丢失**，后续 `RefineArgsBlockId` 等 pass 就无法正确识别控制流结构了。

**本次更新要点**：
1. **新增 `block_id` 定向转移**（`transferBlockIdToInsertedReplacement`）：当带 `ssbuffer.block_id` 的控制流 op 被 canonicalize 替换成**不同名**的新 op 时（旧 `transferAttrs` 受"同名约束"限制无法处理这种情况），把 `block_id` 单独转移给新插入的后继 op；
2. 两个替换回调都挂上新逻辑：`notifyOperationReplaced(op, newOp)` 直接调用；`notifyOperationReplaced(op, values)` 对每个 replacement value 的 defining op 逐一尝试（并去掉了旧的提前 `return`）；
3. **两套转移机制的约束差异**：
   - `transferAttrs`：要求 from/to **同名**且都是控制流 op → 转移**全部**属性；
   - `transferBlockIdToInsertedReplacement`：只要求 from 是控制流 op（**对 to 的类型/名字不设限**）+ to 在 `recentInserts` 中 + 与 from **同 block** → 只转移 `kBlockId`，且不覆盖已有。

---

## 二、核心机制：RewriterBase::Listener

MLIR 的 pattern rewrite 支持注册一个 `Listener`，它会在 rewrite 过程中收到各种通知（op 插入、删除、替换）。这个 Pass 通过自定义 Listener，在 op 被替换/删除时**把旧 op 的关键属性转移到新 op**。

关键回调：
- `notifyOperationInserted`：op 被插入
- `notifyOperationErased`：op 被删除
- `notifyOperationReplaced(op, newOp)`：op 被新 op 替换
- `notifyOperationReplaced(op, values)`：op 被一组 value 替换

---

## 三、逐行讲解

### 3.1 头文件与辅助函数（L23-L55）

```cpp
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                       // scf 方言
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"          // 冻结 pattern 集
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"    // 贪心 rewrite 驱动
#include "ascend/include/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.h"
#include "DynamicCVPipeline/Common/Utils.h"
```

```cpp
static void debugDumpIr(StringRef stage, Operation *op) { ... }

static bool isTrackedControlFlowOp(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static bool canTransferAttrs(Operation *from, Operation *to) {
  return from && to && from != to &&
         isTrackedControlFlowOp(from) && isTrackedControlFlowOp(to) &&
         from->getName() == to->getName();   // 只允许同名 op 之间转移
}
```

- 关键约束：**只在同类型的控制流 op 之间转移属性**（`from->getName() == to->getName()`）。避免把 `scf.for` 的属性错误地贴到 `scf.if` 上。

### 3.2 PreserveControlAttrsListener（L57-L149）

```cpp
class PreserveControlAttrsListener : public RewriterBase::Listener {
public:
  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint) override {
    recentInserts.insert(op);   // 记录最近插入的 op
  }
  void notifyOperationErased(Operation *op) override {
    recentInserts.remove(op);   // 从集合移除
  }
  // 回调 1：op 被单个新 op 替换
  void notifyOperationReplaced(Operation *op, Operation *newOp) override {
    transferAttrs(op, newOp);                          // 全属性转移（同名约束）
    transferBlockIdToInsertedReplacement(op, newOp);   // block_id 定向转移（新增）
  }
  // 回调 2：op 被一组 value 替换
  void notifyOperationReplaced(Operation *op, ValueRange values) override {
    if (Operation *newOp = findReplacementOp(op, values)) {
      transferAttrs(op, newOp);                        // 找到后继 → 全属性转移
    }
    // 新增：对每个 replacement value 的定义 op 尝试 block_id 转移
    //（不再提前 return——即使 findReplacementOp 失败也要继续）
    for (Value value : values) {
      if (!value) continue;
      if (Operation *defOp = value.getDefiningOp()) {
        transferBlockIdToInsertedReplacement(op, defOp);
      }
    }
  }
```

`findReplacementOp`（L87-L117）：当 op 被"一组 value"替换时，需要从这些 value 的 defining op 中找出真正的"后继 op"来承接属性：

```cpp
  Operation *findReplacementOp(Operation *oldOp, ValueRange replacements) const {
    if (!isTrackedControlFlowOp(oldOp)) return nullptr;
    // 优先：从 replacements 的 defining op 中找最近插入且可转移的
    for (Value value : replacements) {
      if (!value) continue;
      Operation *defOp = value.getDefiningOp();
      if (defOp && recentInserts.contains(defOp) && canTransferAttrs(oldOp, defOp))
        return defOp;
    }
    // 兜底：在同一 block 的 recentInserts 里逆序找候选
    Block *oldBlock = oldOp->getBlock();
    if (!oldBlock) return nullptr;
    for (Operation *candidate : llvm::reverse(recentInserts.getArrayRef())) {
      if (!canTransferAttrs(oldOp, candidate)) continue;
      if (candidate->getBlock() != oldBlock) continue;
      return candidate;
    }
    return nullptr;
  }
```

`transferAttrs`（L119-L128）：全属性转移，要求**同名**：

```cpp
  static void transferAttrs(Operation *from, Operation *to) {
    if (!canTransferAttrs(from, to)) return;   // 同名 + 都是控制流 op
    for (NamedAttribute attr : from->getAttrs()) {
      if (to->hasAttr(attr.getName())) continue;   // 目标已有则不覆盖
      to->setAttr(attr.getName(), attr.getValue());
    }
  }
```

- 只转移目标 op **还没有**的属性，避免覆盖已有属性。

`transferBlockIdToInsertedReplacement`（L130-L146，**新增**）：`block_id` 专用的定向转移，**放宽了同名约束**：

```cpp
  void transferBlockIdToInsertedReplacement(Operation *from,
                                            Operation *to) const {
    if (!from || !to || from == to || !isTrackedControlFlowOp(from)) {
      return;   // 只约束 from 是被跟踪的控制流 op，对 to 的类型不设限
    }

    Attribute blockId = from->getAttr(CVPipeline::kBlockId);
    if (!blockId || to->hasAttr(CVPipeline::kBlockId)) {
      return;   // 无 block_id 可转，或目标已有
    }

    if (!recentInserts.contains(to) || from->getBlock() != to->getBlock()) {
      return;   // to 必须是本次 rewrite 新插入的，且和 from 同 block
    }

    to->setAttr(CVPipeline::kBlockId, blockId);
  }
```

- **为什么需要这套新机制**：canonicalize 可能把 `scf.for` 之类的控制流 op 替换成**不同类型**的 op（例如消解成更简单的形式）。旧 `transferAttrs` 的同名约束此时直接放弃，`block_id` 就丢了。而 `block_id` 的语义是"这个 op 位于哪个基本块区域"，**和 op 的具体类型无关**——即使 `scf.for` 变成了别的形态，它占据的位置仍然属于原来的 block，所以值得放宽约束单独转移。
- **安全性靠什么保证**（毕竟放宽了约束）：
  1. `to` 必须在 `recentInserts` 里——只认**本次 canonicalize 过程中新插入**的 op，不会污染 IR 里的既有 op；
  2. `from->getBlock() == to->getBlock()`——位置上确实承接了原 op；
  3. `to` 已有 `kBlockId` 则不覆盖。
- 成员 `recentInserts`（L148）：`llvm::SetVector<Operation *>`，去重且保序（`findReplacementOp` 的兜底逻辑依赖插入顺序的逆序遍历）。

### 3.3 populateCanonicalizationPatterns（L151-L160）

```cpp
static void populateCanonicalizationPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  // 收集所有已加载 dialect 的 canonicalization patterns
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  // 收集所有已注册 op 的 canonicalization patterns
  for (RegisteredOperationName opName : ctx->getRegisteredOperations())
    opName.getCanonicalizationPatterns(patterns, ctx);
}
```

- 加载所有方言和 op 的 canonicalization patterns，做一次全局 canonicalize。

### 3.4 runOnOperation（L164-L187）

```cpp
void mlir::triton::PreserveControlAttrsCanonicalizePass::runOnOperation() {
  if (CVPipeline::hasFallbackAttr(getOperation())) return;

  debugDumpIr("before PreserveControlAttrsCanonicalizePass", getOperation());

  RewritePatternSet patterns(&getContext());
  populateCanonicalizationPatterns(&getContext(), patterns);

  PreserveControlAttrsListener listener;   // 自定义监听器
  GreedyRewriteConfig config;
  config.setListener(&listener);           // 挂载监听器

  // 贪心应用 pattern 直到收敛
  if (failed(applyPatternsGreedily(getOperation(),
                                   FrozenRewritePatternSet(std::move(patterns)),
                                   config))) {
    getOperation()->emitError("PreserveControlAttrsCanonicalizePass failed");
    CVPipeline::setFallbackAttr(getOperation(), CVPipeline::ERRCODE_FAILED);
    return;
  }

  debugDumpIr("after PreserveControlAttrsCanonicalizePass", getOperation());
}
```

- `applyPatternsGreedily`：MLIR 提供的贪心 rewrite 驱动，反复应用 pattern 直到 IR 不再变化（fixpoint）。
- `FrozenRewritePatternSet`：把 pattern 集冻结（只读），避免贪心过程中修改 pattern 集。
- 失败时设置 fallback 属性（回退到非优化路径），符合整条 CV pipeline 的兜底约定。

### 3.5 Pass 注册（L189-L204）

```cpp
std::unique_ptr<OperationPass<ModuleOp>> createPreserveControlAttrsCanonicalizePass() {
  return std::make_unique<PreserveControlAttrsCanonicalizePass>();
}
void registerPreserveControlAttrsCanonicalizePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createPreserveControlAttrsCanonicalizePass();
  });
}
```

---

## 四、算法流程总结

```
1. 加载所有 dialect/op 的 canonicalization patterns
2. 创建自定义 Listener（拦截 op 插入/删除/替换事件）
3. 挂载 Listener 到 GreedyRewriteConfig
4. applyPatternsGreedily 反复 canonicalize 到 fixpoint
   - 控制流 op 被同名新 op 替换 → transferAttrs 转移全部属性
   - 控制流 op 被任意新插入 op（同 block）替换 →
     transferBlockIdToInsertedReplacement 定向转移 block_id
```

---

## 五、面试要点

1. **为什么要保留控制流 op 的属性？**
   前面 pass（`AddBlockIdForControlOps`、`MarkMainLoop`、`DataDependencyAnalysis`）在控制流 op 上打了 `ssbuffer.block_id`、`ssbuffer.main_loop` 等标记，后续 `RefineArgsBlockId` 依赖这些标记。canonicalize 若丢失它们，会导致后续 pass 失效。

2. **为什么用 Listener 而不是"canonicalize 后重新打标"？**
   因为 canonicalize 可能把 op 替换成全新 op，重建映射关系很困难。用 Listener 在替换发生的瞬间"就地转移"属性，是最可靠的方式。

3. **为什么限制"同名 op 才能转移属性"？**
   不同名的 op（如 `scf.for` → `scf.if`）语义不同，其上的属性（如 `main_loop`）含义也不同，盲目转移会产生错误标记。只允许同名 op 转移保证了属性语义一致。

4. **`recentInserts` 集合的作用？**
   当 op 被"一组 value"替换（而非单个新 op）时，需要从最近插入的 op 中推断出真正的后继 op。`recentInserts` 记录 rewrite 过程中新插入的 op，提供候选集来匹配属性承接者。

5. **既然同名才能转移，为什么 block_id 可以跨类型转移？**
   `main_loop` 这类属性绑定 op 的"循环语义"，换 op 类型就变了含义；而 `block_id` 绑定的是**位置**（op 所在的基本块区域），与 op 类型无关。canonicalize 把控制流 op 消解成其他形态时，新 op 占据的位置仍属于原 block，所以 `transferBlockIdToInsertedReplacement` 放宽了"同名"约束，只保留三个安全护栏：目标必须是本次 rewrite 新插入（`recentInserts`）、必须与原 op 同 block、目标没有自己的 `block_id`。这是"属性语义决定转移策略"的典型设计。

6. **为什么 `notifyOperationReplaced(op, values)` 不能在找不到后继时提前返回？**
   旧逻辑找到 `findReplacementOp` 的同名候选就 return，找不到就什么都不做。新逻辑即使全属性转移失败，也要遍历每个 replacement value 的 defining op 尝试 `block_id` 转移——两套机制的判定条件不同（同名 vs 同 block+新插入），前者失败不代表后者也失败。
