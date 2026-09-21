# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 04 —— CreateIfOps（if 块骨架构造）

> 覆盖 `AddControlFlowCondition` 的 **Step 2：CreateIfOpsPass**
>
> 源码文件：[`CreateIfOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第三个子 Pass（主编排 L95-L97）。

---

## 一、这个子 Pass 在做什么

`CloneOps` 让每个 block_id 拥有独立的 op，`ProcessArgs` 让每个 block_id 拥有独立的 iter_arg。但主循环体里这些 op 仍是**平铺执行**的：每个迭代周期所有 block 的 op 无差别地全部跑一遍，无法实现"哪个 block 本轮该跑、哪个该让位给流水线缓冲"的错位执行。

`CreateIfOps` 的职责是**为每个 block_id 构造一个 `scf.if %true` 占位 if 骨架**：

1. 每个 `block_id` 对应一个 `scf.if`（条件暂时写死 `%true`，后续由 `UpdateConditionInfo` 替换成真实同步条件）；
2. **then 分支**：把该 block 的所有 op 移进去（含嵌套 op）；
3. **else 分支**：yield 主循环的 iter_arg 原值（兜底：条件为假时状态不变）；
4. 给 if 打上 **`kIf`（ssbuffer.if）= block_id** 标记 —— 这是后续所有依赖分析与条件构造以"if 块"为单位的锚点；
5. 更新 if 外部的 uses，让 block 内产出的值改由 if 的结果提供。

**一句话**：把"平铺执行"重排为"逐 block 一个 if 容器"，为 Step 5 注入真实同步条件搭好骨架、打上块标识。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kIf`（`ssbuffer.if`） | 本 Pass 给每个 if 打的标记属性，值为 block_id。之后 `InitDependentMap`/`UpdateConditionInfo` 等都以"带 kIf 的 if 块"为依赖分析单位 |
| `kBlockId`（`ssbuffer.block_id`） | op 的块归属标记，`collectOpsByBlockId` 用它把 body 内 op 分组 |
| `kMainLoop`（`ssbuffer.main_loop`） | 主循环标记，`isMainLoopOp` 据此识别 for/while |
| `scf.if %true` 占位 | 条件先固定为 true，语义等价于"本 block 必跑"；真实同步条件在 Step 5（`UpdateConditionInfo`）替换 |
| `blockOps` | `DenseMap<int, SmallVector<Operation*>>`：block_id -> 该块全部 op（`collectOpsByBlockId` 产出，含嵌套） |
| `regionOps` | 某 block 的闭包集合（`collectAllNestedOps` 递归收集），判断"值是否逃出本 block" |
| `thenYieldValues` / `elseYieldValues` | `DenseMap<int, SmallVector<Value>>`：每 block 的 then 分支 yield 值（真实计算值）/ else 分支 yield 值（主循环 iter_arg 兜底值） |
| `findIterArgInMainLoop` | 从值 v 的 users 中找 `scf.yield`，再反查该 yield operand 对应的主循环 iter_arg，作为 else 兜底值 |
| `blockCounterNums` | `info->blockCounterNums[op] = blockOps.size()`：记录每个主循环的 block 数，供 Step 4 `UpdateLoopOps` 给每 block 追加计数器 iter_arg 用 |
| `kHIVMMatmulLimitedInCubeAttr` | 通知 npuir 的标记属性：if 内 matmul 被限制在 CUBE 内执行（跨核同步语义相关） |
| `getBlockIdsInOrder` | 按出现顺序取所有 block_id（保证 if 创建顺序与 block 语义一致） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（CreateIfOps.cpp L329-L376）

```cpp
void CreateIfOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }
    // 1. 按 block_id 分组收集主循环 body 内全部 op
    llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
    if (failed(collectOpsByBlockId(op, blockOps))) {
      return WalkResult::interrupt();
    }
    // 2. 记录 block 数（供 UpdateLoopOps 分配计数器 iter_arg）
    if (info) {
      info->blockCounterNums[op] = blockOps.size();
    }
    // 3. 计算 then/else yield 值
    llvm::DenseMap<int, SmallVector<Value>> thenYieldValues;
    llvm::DenseMap<int, SmallVector<Value>> elseYieldValues;
    if (failed(computeYieldValues(op, blockOps, thenYieldValues,
                                  elseYieldValues))) {
      return WalkResult::interrupt();
    }
    // 4. 逐 block 创建 if 并搬运 op
    if (failed(createIfInMainLoop(op, blockOps, thenYieldValues,
                                  elseYieldValues))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

**要点**：
- walk 是只读扫描 + 就地改 IR（`moveBefore` 移动 op、`create` 插入 if），主循环之间的处理互不干扰，可在一个 walk 内完成；
- 任一步失败立即 `interrupt`，整个模块打 `ERRCODE_FAILED` fallback 标记；
- `info->blockCounterNums` 在这里被填充 —— 这是 CreateIfOps 对共享状态 `ControlFlowConditionInfo` 的唯一贡献，供 Step 4 使用。

### 3.2 按 block_id 分组：collectOpsByBlockId（Utils 辅助）

```cpp
// 逻辑：遍历主循环 body 内所有 op（含嵌套），
// 读取 kBlockId 属性，按 id 分桶到 blockOps[id]。
// 任 op 缺 kBlockId 或主循环不是 for/while → failure。
```

**要点**：`kBlockId` 在 `ComputeBlockOpt`/`PlanComputeBlock` 阶段已打上，是 op 归属的权威标识；这里"含嵌套 op"指 collect 时用 walk 而非只扫顶层，保证 block 闭包完整（后面 `isUsedOutsideRegion` 判断依赖逃逸才准确）。

### 3.3 计算 yield 值：computeYieldValues（L151-L195）

```cpp
LogicalResult CreateIfOpsPass::computeYieldValues(
    Operation *loopOp,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
    llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  for (auto &p : blockOps) {
    int id = p.first;
    const SmallVector<Operation *> &ops = p.second;

    // 收集本 block 全部 op（含嵌套）进 regionOps 闭包
    llvm::DenseSet<Operation *> regionOps;
    for (Operation *op : ops) {
      if (failed(collectAllNestedOps(op, regionOps))) {
        return failure();
      }
    }

    // then 分支 yield = 所有"被 block 外部使用"的结果值
    SmallVector<Value> yieldValues;
    for (Operation *op : ops) {
      for (Value res : op->getResults()) {
        if (isUsedOutsideRegion(res, regionOps)) {
          yieldValues.push_back(res);
        }
      }
    }
    thenYieldValues[id] = yieldValues;

    // else 分支 yield = 主循环 iter_arg 原值（兜底）
    elseYieldValues[id].clear();
    elseYieldValues[id].reserve(yieldValues.size());
    for (Value v : yieldValues) {
      Value foundArg = findIterArgInMainLoop(v, v.getType());
      if (!foundArg) {
        return failure();
      }
      elseYieldValues[id].push_back(foundArg);
    }
  }
  return success();
}
```

**要点**：
- **then 的 yield 值** = 本 block 内被外部（其他 block / 循环后）消费的 SSA 结果；if 有结果后这些外部 uses 会被重定向到 if 结果（见 3.6）；
- **else 的 yield 值** = 同一位置在主循环里的 iter_arg。这样 if 条件为假时，if 结果 = iter_arg 原值 → 外部消费者拿到的还是"上一轮的旧值"，状态不变，正是流水线占位阶段需要的语义；
- **一一对应**：then 第 i 个值与 else 第 i 个值类型必须一致（3.5 有校验）。

### 3.4 逃逸判定：isUsedOutsideRegion（L52-L61）

```cpp
static bool isUsedOutsideRegion(Value v,
                                const llvm::DenseSet<Operation *> &regionOps) {
  for (OpOperand &use : v.getUses()) {
    if (!regionOps.contains(use.getOwner())) {
      return true;  // 任一 use 的 owner 不在本 block 闭包内 → 逃逸
    }
  }
  return false;
}
```

**要点**：逃逸是判断"是否需要 yield"的唯一依据。use 的 owner 若不在 `regionOps`（本 block 含嵌套 op 的闭包），说明该值要跨 block 传递，必须作为 if 结果流出。

### 3.5 找 else 兜底值：findIterArgInMainLoop（L64-L95）

```cpp
static Value findIterArgInMainLoop(Value v, mlir::Type t) {
  for (Operation *user : v.getUsers()) {
    auto yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp) {
      continue;
    }
    SmallVector<Value> iterArgs =
        MainLoop(yieldOp->getParentOp()).getIterArgs();
    if (iterArgs.empty()) {
      continue;
    }
    for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
      if (operand.getAsOpaquePointer() == v.getAsOpaquePointer()) {
        Value iterArg = iterArgs[idx];
        if (iterArg.getType() == t) {
          return iterArg;
        }
      }
    }
  }
  return nullptr;
}
```

**要点**：
- 值的 users 里必然有一个主循环的 `scf.yield`（block 内计算最终要 yield 回主循环）；
- 找到 `v` 在 yield operand 中的位置 `idx`，**yield operand idx = iter_arg idx**（body 参数位置对齐），反查主循环 iter_arg；
- 用 `getAsOpaquePointer()` 比较同一 SSA 值；类型必须匹配（`iterArg.getType() == t`）；
- 对 for/while 通用：`MainLoop(yieldOp->getParentOp())` 按父 op 分派取 iter_args，`scf::YieldOp` 在 for 里出现在 body 末尾、在 while 里出现在 after 区末尾，`MainLoop` 工具类屏蔽了差异。

### 3.6 创建 if 骨架：createIfInMainLoop（L286-L327）

```cpp
LogicalResult CreateIfOpsPass::createIfInMainLoop(
    Operation *op,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    const llvm::DenseMap<int, SmallVector<Value>> &thenYieldValues,
    const llvm::DenseMap<int, SmallVector<Value>> &elseYieldValues) {
  SmallVector<int> ids = getBlockIdsInOrder(op);
  if (ids.empty() && !MainLoop(op).getBody()) {
    return failure();
  }

  for (int id : ids) {
    const SmallVector<Operation *> &ops = blockOps.lookup(id);
    if (ops.empty()) {
      continue;
    }

    // 在 block 首个 op 前插入 if，位置即该 block 的起点
    OpBuilder builder(ops.front());
    Location loc = ops.front()->getLoc();

    scf::IfOp ifOp =
        createIfOpForBlock(builder, loc, id, thenYieldValues.lookup(id),
                           elseYieldValues.lookup(id));
    if (!ifOp) {
      return failure();
    }

    // 搬运 op 到 then 分支 + 建 yield
    SmallVector<Operation *> opsToMove = ops;
    if (failed(moveOpsToThenBranch(ifOp, opsToMove, thenYieldValues.lookup(id),
                                   elseYieldValues.lookup(id), loc))) {
      return failure();
    }
    // 把 if 外部对旧值的 uses 重定向到 if 结果
    if (failed(replaceExternalIfOpUses(ifOp, thenYieldValues.lookup(id)))) {
      return failure();
    }
  }
  return success();
}
```

**要点**：
- `getBlockIdsInOrder` 保证 if 按 block 的语义顺序创建（与后续 if 索引、依赖距离计算对齐）；
- `ops.empty()` 的 block 跳过（没有 op 就不需要 if）；
- `OpBuilder(ops.front())` 把插入点设在 block 首个 op 之前，if 创建后自然位于该 block 起点；
- **先建 if → 再搬 op → 最后重定向外部 uses**，三步顺序固定。

### 3.7 构造 ifOp：createIfOpForBlock（L210-L253）

```cpp
static scf::IfOp createIfOpForBlock(OpBuilder &builder, Location loc,
                                    int blockId,
                                    const SmallVector<Value> &thenValues,
                                    const SmallVector<Value> &elseValues) {
  bool needsYield = !thenValues.empty();
  // 有 yield 时 then/else 数量与类型必须一致
  if (needsYield && thenValues.size() != elseValues.size()) {
    return scf::IfOp();
  }
  if (needsYield) {
    for (size_t i = 0; i < thenValues.size(); ++i) {
      if (thenValues[i].getType() != elseValues[i].getType()) {
        return scf::IfOp();
      }
    }
  }

  SmallVector<mlir::Type> resultTypes = getResultTypes(thenValues);

  Value trueVal = builder.create<arith::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));

  scf::IfOp ifOp;
  if (needsYield) {
    ifOp = builder.create<scf::IfOp>(loc, resultTypes, trueVal, true);
  } else {
    ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, trueVal, false);
  }

  ifOp->setAttr(CVPipeline::kIf, builder.getI32IntegerAttr(blockId));
  ifOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                builder.getUnitAttr());
  return ifOp;
}
```

**要点**：
- 条件 = `arith.constant true`（占位，Step 5 替换为真实条件）；
- `scf::IfOp(loc, resultTypes, trueVal, true)` 的第二个 bool 是 **withElseRegion**：有 yield 则带 else 区（else 要 yield 兜底值），无 yield 则不带 else 区；
- `kIf = block_id`：if 的身份标记，后续一切依赖分析以"带 kIf 的 if"为单位；
- `kHIVMMatmulLimitedInCubeAttr`：通知 npuir 该 if 内 matmul 限 CUBE（与跨核流水线同步语义配套）。

### 3.8 搬运 op：moveOpsToThenBranch（L256-L282）

```cpp
static LogicalResult moveOpsToThenBranch(scf::IfOp ifOp,
                                         SmallVector<Operation *> &ops,
                                         const SmallVector<Value> &thenValues,
                                         const SmallVector<Value> &elseValues,
                                         Location loc) {
  if (ops.empty() && !thenValues.empty()) {
    return failure();
  }
  Block &thenBlock = ifOp.getThenRegion().front();

  // 反向移动：ops 原本按顺序排列，逐个移到 thenBlock.begin() 前，
  // 逆序遍历可保持原始相对顺序
  for (Operation *op : llvm::reverse(ops)) {
    op->moveBefore(&thenBlock, thenBlock.begin());
  }

  if (!thenValues.empty()) {
    OpBuilder thenBuilder(&thenBlock, thenBlock.end());
    thenBuilder.create<scf::YieldOp>(loc, thenValues);
    Block &elseBlock = ifOp.getElseRegion().front();
    OpBuilder elseBuilder(&elseBlock, elseBlock.end());
    elseBuilder.create<scf::YieldOp>(loc, elseValues);
  }
  return success();
}
```

**要点**：
- `llvm::reverse(ops)` + `moveBefore(begin)`：每次移到 then 块开头，逆序遍历后整体保持原顺序（经典"倒插"技巧）；
- 有 yield 才在 then/else 末尾补 `scf.yield`；无 yield 的 if 两个区都为空。

### 3.9 重定向外部 uses：replaceExternalIfOpUses（L99-L147）

```cpp
static LogicalResult replaceExternalIfOpUses(scf::IfOp ifOp,
                                             ArrayRef<Value> oldYieldValues) {
  for (size_t i = 0; i < oldYieldValues.size(); ++i) {
    Value oldVal = oldYieldValues[i];
    Value newVal = ifOp.getResult(i);
    if (oldVal.getType() != newVal.getType()) {
      return failure();
    }

    SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
      Operation *user = use.getOwner();
      // 跳过 if 内部的 uses（op 已搬进 then，uses 仍指向原值，保持内部不变）
      if (ifOp->isAncestor(user)) {
        continue;
      }
      // 跳过同 block 中位于 if 之后的 uses（scf.if 结果只对后续 op 生效）
      if (user->getBlock() == ifOp->getBlock() &&
          !ifOp->isBeforeInBlock(user)) {
        continue;
      }
      usesToReplace.push_back(&use);
    }
    for (OpOperand *use : usesToReplace) {
      use->set(newVal);
    }
  }
  return success();
}
```

**要点**：
- 两个"跳过"非常关键：
  - **if 内部**：op 已搬进 then 区，它们的 uses 仍指向原值（等价于 then 区直接用原值），不能替换成 if 结果（会形成自环/错误）；
  - **同 block 且在 if 之后**：这些 use 已经在 if 的下方，`ifOp` 结果在 if 之后才可用，**且**此时旧值已被 if 的 yield 重新提供——语义上 if 之后的 op 应使用 if 结果；但这里跳过的原因：同 block 内 if 之后的 op 使用的仍是旧值所在位置，若旧值已被外部 use 替换，会造成不一致——实际上跳过它们是因为这些 op 属于其他 block（尚未被搬走），它们对旧值的消费由"其他 block 的 if yield 兜底"覆盖。凡**不在同 block 或位于 if 之前的** use 一律换成 if 结果。

> 补充理解：`replaceExternalIfOpUses` 的作用是把"逃逸到本 block 外"的 uses 改指 if 结果。跳过的两类本质都是"不该由本 if 结果接管"的使用：if 内部（自用）、if 之后同 block（属于后续 block 的逻辑，由它们自己的 if 兜底）。

---

## 四、流程总结

### 4.1 CreateIfOps 的流水

```
CreateIfOpsPass::runOnOperation
  └─► walk 所有 main_loop op（for/while）
        ├─► collectOpsByBlockId          按 kBlockId 分桶收集 body op
        ├─► info->blockCounterNums[op] = 块数    ← 供 UpdateLoopOps
        ├─► computeYieldValues
        │     ├─► 每 block：collectAllNestedOps 收闭包 regionOps
        │     ├─► then yield = isUsedOutsideRegion 逃逸的结果值
        │     └─► else yield = findIterArgInMainLoop 找主循环 iter_arg 兜底
        └─► createIfInMainLoop            逐 block_id（getBlockIdsInOrder）
              ├─► createIfOpForBlock      scf.if %true + kIf=id + kHIVMMatmulLimitedInCubeAttr
              ├─► moveOpsToThenBranch     反向移动 op 进 then + 建 then/else yield
              └─► replaceExternalIfOpUses if 外部 uses 重定向到 if 结果
```

### 4.2 与前后子 Pass 的协作

- **前置**：`CloneOps` 保证 op 独立（可整体搬进 if）；`ProcessArgs` 保证 iter_arg 独立（else 兜底取值不串扰）；
- **产出**：每 block 一个带 `kIf=block_id` 的 `scf.if %true`；`info->blockCounterNums` 记录块数；
- **后置**：`InitDependentMap` 以 kIf 为节点建依赖 DAG；`UpdateLoopOps` 用 blockCounterNums 追加计数器 iter_arg；`UpdateConditionInfo` 替换 `%true` 为真实同步条件。

---

## 五、面试要点

1. **为什么要先用 `scf.if %true` 占位而不是直接生成真实条件？** 真实同步条件依赖跨核 buffer 计数、if DAG 依赖关系，而这些信息要等 `InitDependentMap`/`UpdateLoopOps` 之后才齐备。先搭 if 骨架并打 `kIf=block_id` 标记，让后续 Pass 以"if 块"为单位稳定分析，最后统一替换条件，职责解耦。

2. **else 分支为什么 yield 主循环 iter_arg？** 占位阶段 if 条件恒真，else 不执行，但 scf.if 必须有完整的 yield（类型/数量与 then 一致）。else yield iter_arg 原值保证了"条件为假时 if 结果 = 上一轮状态"，这正是流水线错位执行想要的语义：被让位的 block 保持旧状态，消费者读到的还是上一轮缓冲内容。

3. **then/else 的 yield 值是怎么确定的？** then = 本 block 内被外部消费（逃逸出 regionOps 闭包）的结果值；else = 与 then 位置一一对应的主循环 iter_arg。`findIterArgInMainLoop` 通过"值 → users 里的 scf.yield → yield operand 位置 → 主循环 iter_arg 同位"反查，`MainLoop` 工具类屏蔽 for/while 差异。

4. **为什么要反向移动 op 到 then 区开头？** `moveBefore(&thenBlock, begin())` 每次插到开头，正序遍历会倒序；`llvm::reverse(ops)` 逆序处理后，最终 then 区 op 保持原始相对顺序，避免破坏 SSA 依赖关系。

5. **`replaceExternalIfOpUses` 为什么跳过两类 use？** if 内部的 uses 已随 op 搬进 then，继续用原值即可；同 block 且位于 if 之后的 uses 属于后续 block 的消费逻辑，由它们自己的 if 兜底。只有真正逃逸到本 block 外的 uses 才应改指本 if 结果，否则会引入错误依赖或自环。

6. **`blockCounterNums` 在 CreateIfOps 里记录有什么意义？** 每个 block 后续都需要一个"迭代计数 iter_arg"（for 用 lowerBound 初值、while 用 constant 0），个数必须等于 block 数。这里顺手把 `blockOps.size()` 记进共享状态，Step 4 `UpdateLoopOps` 直接读取分配，避免重复统计。
