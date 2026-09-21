# SplitDataflow 子 Pass 逐行讲解（六）：SeparateCVScope

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/SeparateCVScope.cpp`
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 5 步**

---

## 一、这个 Pass 在做什么

把每个 function 的 body **克隆成两个 `scope.scope`**：一个标记为 `VECTOR` 核、一个标记为 `CUBE` 核。然后在每个 scope 内部，**删除不属于本核的 op**，并**归一化控制流**（把不属于本核的 yield/结果替换成"中性值"）。最后做一次 **Mechanism A 冗余清理**：消除 VECTOR scope 里"自己存自己读"的 store→load 对。

这实现了 CV 流水线的关键一步：**把原本混合在一起的 Cube 计算和 Vector 计算，物理地分离到两个独立的核 scope 中**，为后续各自的编译和并行执行做准备。

**本次更新要点**：
1. **worklist 传递查找**：`findLiveUser`/`findNonTermUser` 从"只看直接 user"升级为 `findScopeRelevantUser`——沿 use-def 链**传递**跟踪（穿过 yield→loop result 的"纯转发"链），判断一个 value 在本 scope 里是否真的有活跃使用者。旧的直接判断在嵌套循环/多级 yield 场景下会误判（把"只剩转发"当成"还有人用"）；
2. **循环携带槽位活跃性分析**：新增 `needsLoopCarryPreserve`（同时支持 `scf.for` 和 `scf.while`）+ `slotHasActiveUse`，精确判定哪个 iterArg 槽位必须在本 scope 保留，而不是一刀切；
3. **terminator 归一化重构**：旧 `neutralizeYieldInRegion` → 新 `neutralizeRegionTerminators`，统一处理 `scf.yield` **和** `scf.condition`（while 的 condition 里 operand 0 是条件本身，携带参数从 operand 1 开始）；
4. **Mechanism A 冗余清理（新增）**：`replaceRedundantVectorStoreLoad` 按 `ssbuffer.transfer_id` 识别 VECTOR scope 里冗余的 store→load 对，用存储值直接替换 load 结果，同时清除 load 上的 `annotation.mark volatile` 标记；
5. **控制流 op 集合扩展**：`isControlFlowOp` 新增 `scf::ParallelOp`；
6. **循环结果"死壳"保留逻辑**：`normalizeRegionOp` 新增"循环体内无本核内容但结果仍被本核消费 → 整个循环保留"的分支，避免误中性化改变语义。

---

## 二、关键背景

| 概念 | 说明 |
|---|---|
| `scope::ScopeOp` | BiSheng 的 scope dialect，用于划分"作用域"（这里一个 scope = 一个核） |
| `hivm::TCoreTypeAttr` | 核类型属性，取值 `CUBE` / `VECTOR` |
| `ssbuffer.core_type` | 每个 op 上的核归属标记（"CUBE"/"VECTOR"，多结果 op 可能逗号分隔） |
| "中性值"（neutral value） | 用 0（或空 memref）填充不属于本核的 op 结果，保持 IR 结构完整 |
| `ssbuffer.transfer_id` | `InterCoreTransferAndSync` 给每次传输打的编号，Mechanism A 用它配对 store/load |
| `annotation::MarkOp` + `kMemrefExtVolatile` | 标量读的 volatile 注解，只在 CUBE 侧的 load 存活时有意义 |

核心难点：op 被拆到两个 scope 后，**原本的 SSA use-def 链会被打断**。所以需要"中性化"——在某个 scope 里，一个 value 如果"本不该存在"，就用中性值占位，保证 scope 内部 SSA 仍然合法。

第二个难点（本次更新的核心动机）：判断"还有没有人用"不能再只看**直接 user**。一个 value 可能经过 `yield → 循环 result → 又被 yield → ...` 的**纯转发链**，最后才落到真正属于本核的消费者上。必须把这条链走完（worklist 传递），才能区分"真活跃"和"纯转发后死亡"。

---

## 三、逐行讲解

### 3.1 核心类型信息解析（L79-L153）

```cpp
struct CoreTypeInfo {
  SmallVector<StringRef> resultTypes;
  StringRef getResultType(size_t index) const { ... }        // 越界取 front()
  bool hasResultForScope(StringRef scopeType) const { ... }  // 任一 result 属于该核
  bool allResultsMatchScope(StringRef scopeType, unsigned numResults) const { ... }
};

static std::optional<CoreTypeInfo> parseCoreTypeInfo(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
  if (!attr) return std::nullopt;
  StringRef raw = attr.getValue();
  SmallVector<StringRef> parts;
  raw.split(parts, ',', -1, false);     // "CUBE, VECTOR" → ["CUBE", " VECTOR"]
  for (auto &part : parts) part = part.trim();
  info.resultTypes.assign(parts.begin(), parts.end());
  return info;
}
```

- `ssbuffer.core_type` 可能是单个 `"CUBE"` 或逗号分隔的 `"CUBE, VECTOR"`（多结果 op，每个 result 可能属于不同核）。

`getScopeMatchInfo`（L139-L149）：返回两 bit 信息——`matches`（op 是否和该核有任何关系）和 `allResultsMatch`（是否所有结果都属于该核）。二者组合出三种情况：完全属于 / 完全不属于 / 部分属于。

### 3.2 构建中性值（L155-L181）

```cpp
static Value buildNeutralValue(OpBuilder &builder, Value oldOperand,
                               Location loc, StringRef scopeType) {
  Type type = oldOperand.getType();
  Operation *createdOp = nullptr;
  if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    createdOp = builder.create<memref::AllocOp>(loc, memrefTy);   // memref → 空缓冲
  } else if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasStaticShape() && !isa<MemRefType>(type)) {
      auto zeroDense = DenseElementsAttr::get(shapedTy, elemZero); // tensor → 全 0
      createdOp = builder.create<arith::ConstantOp>(loc, type, typedAttr);
    }
  } else if (Attribute zeroAttr = builder.getZeroAttr(type)) {
    createdOp = builder.create<arith::ConstantOp>(loc, type, typedZero); // 标量 → 0
  }
  return createdOp ? createdOp->getResult(0) : Value();
}
```

- 根据类型生成"中性值"：memref 用 `alloc`（空缓冲），tensor 用全 0 常量，标量用 0 常量。

### 3.3 createTwoFullScopes：克隆成两个 scope（L183-L232）

这是最核心的拆分函数：

```cpp
static FailureOr<std::pair<scope::ScopeOp, scope::ScopeOp>>
createTwoFullScopes(func::FuncOp funcOp) {
  // 只支持单 block function
  if (!llvm::hasSingleElement(funcOp.getBody())) { ... return failure(); }

  Block &lastBlock = funcOp.getBody().back();
  // 收集 body 中所有非 terminator op
  SmallVector<Operation *> opsToMove;
  for (Operation &op : lastBlock.without_terminator()) opsToMove.push_back(&op);
  if (opsToMove.empty()) return std::make_pair(scope::ScopeOp(), scope::ScopeOp());

  // 在 block 末尾创建第一个 scope（VECTOR），把所有 op 移进去
  Operation *lastOpToMove = opsToMove.back();
  OpBuilder builder(&lastBlock, ++lastOpToMove->getIterator());
  auto vecScope = builder.create<scope::ScopeOp>(builder.getUnknownLoc(), ArrayRef<Type>{});
  vecScope.getBodyRegion().emplaceBlock();
  Block *vecBlock = &vecScope.getBodyRegion().front();
  OpBuilder vecBuilder(vecBlock, vecBlock->end());
  for (Operation *op : opsToMove) { op->remove(); vecBuilder.insert(op); }
  vecBuilder.create<scope::ReturnOp>(...);   // 补上 return

  // 克隆 vecScope 得到 cubeScope
  builder.setInsertionPointAfter(vecScope);
  IRMapping mapping;
  auto cloned = builder.clone(*vecScope.getOperation(), mapping);

  // 分别打上核类型标记
  vecScope->setAttr(hivm::TCoreTypeAttr::name, vecAttr);   // VECTOR
  cubeScope->setAttr(hivm::TCoreTypeAttr::name, cubeAttr); // CUBE
  return std::make_pair(vecScope, cubeScope);
}
```

**关键点**：先无脑地把整个 body 复制两份（一份 VECTOR scope、一份 CUBE scope），两份内容暂时完全一样。接下来才是"裁剪"——在每个 scope 里删掉不属于本核的 op。

### 3.4 动作收集 collectActionsInRegion（L261-L303）

```cpp
enum class PendingActionKind { EraseDirectly, NormalizeControlFlow };

static bool isControlFlowOp(Operation *op) {          // 本次更新：+ ParallelOp
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static void collectActionForOp(Operation *op, StringRef scopeType,
                               SmallVector<PendingAction> &actions) {
  ScopeMatchInfo matchInfo = getScopeMatchInfo(op, scopeType);
  if (!matchInfo.matches) {
    actions.push_back({op, PendingActionKind::EraseDirectly});      // 完全不属于 → 删
  } else if (!matchInfo.allResultsMatch) {
    actions.push_back({op, PendingActionKind::NormalizeControlFlow}); // 部分属于 → 归一化
  }
}

static SmallVector<PendingAction> collectActionsInRegion(Region &region, StringRef scopeType) {
  SmallVector<PendingAction> actions;
  for (Block &block : region)
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>()) continue;
      if (op.getNumRegions() > 0) {
        if (isControlFlowOp(&op)) {
          if (!isPureForScope(&op, scopeType))                  // 控制流且不纯净
            actions.push_back({&op, NormalizeControlFlow});     // → 归一化
        } else if (!matchesScope(&op, scopeType)) {
          actions.push_back({&op, EraseDirectly});              // 带 region 但不匹配 → 删
        }
        continue;
      }
      collectActionForOp(&op, scopeType, actions);
    }
  return actions;
}
```

- `isPureForScope`（L234-L259）：递归检查控制流 op 的所有 region 内的 op 是否**全部**属于本核。是 → 这个控制流整体保留不动；否 → 需要归一化。
- 两种动作：
  - `EraseDirectly`：op 完全不属于本核 → 直接删除（前提是没有存活用户）；
  - `NormalizeControlFlow`：op 部分属于本核（或控制流混合）→ 需要"归一化"（把不属于本核的 yield/结果中性化）。

### 3.5 【本次更新核心】worklist 传递查找：findScopeRelevantUser（L318-L547）

这是本次更新最核心的改动。旧版 `findLiveUser` 只检查 value 的**直接 user**；新版沿 use-def 链**传递**搜索，穿过"纯转发"节点（yield → 循环 result），找到真正 relevant 的消费者。

#### 3.5.1 hasActiveNestedLoopCarryUse（L318-L339）

```cpp
// 该 use 是否对应一个"必须在本 scope 存活"的循环携带槽位
static bool hasActiveNestedLoopCarryUse(OpOperand &use, StringRef scopeType) {
  auto loopOp = dyn_cast<LoopLikeOpInterface>(use.getOwner());
  if (!loopOp) return false;

  auto inits = loopOp.getInitsMutable();      // 循环 init 操作数区间
  unsigned operandIndex = use.getOperandNumber();
  unsigned initsBegin = inits.front().getOperandNumber();
  unsigned initsEnd = initsBegin + inits.size();
  if (operandIndex < initsBegin || operandIndex >= initsEnd) return false;

  unsigned resultIndex = operandIndex - initsBegin;
  return resultIndex < loopOp->getNumResults() &&
         needsLoopCarryPreserve(loopOp.getOperation(), resultIndex, scopeType);
}
```

- 用 `LoopLikeOpInterface` 统一定位 init 槽位区间（for 的 operand 3 起、while 的 operand 0 起，接口屏蔽差异）。

#### 3.5.2 canSkipForwardingUse：识别"纯转发"use（L342-L376）

```cpp
// 跳过只喂给"出 scope 的循环结果槽位"的纯转发 use
static bool canSkipForwardingUse(OpOperand &use, StringRef scopeType) {
  Operation *user = use.getOwner();
  auto info = parseCoreTypeInfo(user);
  if (!info) return false;

  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    unsigned operandIndex = use.getOperandNumber();
    if (operandIndex < kForOpOperandPrefixCount) return false;  // lb/ub/step 不是转发
    unsigned resultIndex = operandIndex - kForOpOperandPrefixCount;
    return resultIndex < forOp.getNumResults() &&
           info->getResultType(resultIndex) != scopeType &&     // 该结果不属于本核
           !needsLoopCarryPreserve(forOp, resultIndex, scopeType); // 且无活跃携带
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {            // 新增：while 分支
    unsigned slotIdx = use.getOperandNumber();
    if (slotIdx >= whileOp.getNumResults()) return false;
    if (info->getResultType(slotIdx) == scopeType) return false;
    if (needsLoopCarryPreserve(whileOp, slotIdx, scopeType)) return false;
    return llvm::none_of(whileOp->getUsers(), [&](Operation *user) {
      return matchesScope(user, scopeType);   // while 的结果完全没有本核消费者
    });
  }
  return false;
}
```

- **for 的结构**：operand 0/1/2 是 lb/ub/step（`kForOpOperandPrefixCount = 3`），之后才是 init。use 落在 init 区且对应 result 不属于本核、也不需要保留携带 → 这个 use 只是"路过"，可以跳过。
- **while 的结构**：operand 从 0 开始就是 init（while 没有 lb/ub/step）。

#### 3.5.3 shouldPreserveFromYield / handleYieldUse（L380-L438）

```cpp
// 只有当"循环携带证据 + 混合核的兄弟结果"同时成立才保留 yield→result 追踪
static bool shouldPreserveFromYield(Operation *yieldOwner, unsigned operandIndex,
                                    StringRef scopeType) {
  bool preserveLoopCarry = needsLoopCarryPreserve(yieldOwner, operandIndex, scopeType);
  if (!preserveLoopCarry) return false;

  unsigned numResults = yieldOwner->getNumResults();
  return llvm::any_of(llvm::seq<unsigned>(0, numResults), [&](unsigned i) {
    return i != operandIndex && ownerInfo->getResultType(i) == scopeType;
    // 有兄弟 result 属于本核 → 这个 while/for 是"混合核"控制流，不能简化
  });
}

// 处理 scf.yield use：传递追踪 result 槽位，或判定 owner 活跃
static Operation *handleYieldUse(scf::YieldOp yieldOp, OpOperand &use,
                                 StringRef scopeType, bool ignoreTerminators,
                                 SmallVector<Value> &worklist) {
  Operation *yieldOwner = yieldOp->getParentOp();
  unsigned operandIndex = use.getOperandNumber();

  if (operandIndex < yieldOwner->getNumResults()) {
    if (shouldPreserveFromYield(yieldOwner, operandIndex, scopeType)) {
      return yieldOwner;                       // 需要保留 → owner 是活跃用户
    }
    worklist.push_back(yieldOwner->getResult(operandIndex));  // 否则继续传递
    return nullptr;
  }
  ...
}
```

- **为什么需要"兄弟 result 判断"**：如果一个 for/while 的部分 result 属于本核、部分不属于，这个循环本身就必须在本核保留（结构上不能拆），此时 yield 到它的 use 视为活跃。

#### 3.5.4 classifyScopeRelevantUse：单个 use 的分类（L470-L518）

```cpp
static Operation *classifyScopeRelevantUse(OpOperand &use, StringRef scopeType,
                                           bool ignoreTerminators,
                                           SmallVector<Value> &worklist) {
  Operation *user = use.getOwner();
  if (!user || !user->getBlock() || canSkipForwardingUse(use, scopeType))
    return nullptr;                              // 纯转发 → 跳过

  if (hasActiveNestedLoopCarryUse(use, scopeType))
    return acceptUser(user, ignoreTerminators);  // 活跃循环携带 → user 活跃

  if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {   // while 的 condition
    Operation *parentOp = conditionOp.getParentOp();
    if (parentOp && matchesScope(parentOp, scopeType))
      return acceptUser(parentOp, ignoreTerminators);
    if (parentOp && use.getOperandNumber() == 0 &&            // 条件本身
        controlFlowOpHasScopeContent(parentOp, scopeType))    // while 体内有本核内容
      return acceptUser(parentOp, ignoreTerminators);
    return nullptr;
  }

  if (auto yieldOp = dyn_cast<scf::YieldOp>(user))
    return handleYieldUse(yieldOp, use, scopeType, ignoreTerminators, worklist);

  if (matchesScope(user, scopeType))
    return acceptUser(user, ignoreTerminators);  // 直接匹配 → 活跃

  if (user->getNumResults() == 0) {
    if (isControlFlowOp(user))                   // 无结果的控制流 → 活跃
      return acceptUser(user, ignoreTerminators);
    return nullptr;
  }

  if (isControlFlowGatingUse(use) &&             // for/parallel 边界或 if 条件
      controlFlowOpHasScopeContent(user, scopeType))
    return acceptUser(user, ignoreTerminators);

  // 其余：把 user 的所有 result 压栈，继续传递查找
  for (Value result : user->getResults())
    worklist.push_back(result);
  return nullptr;
}
```

- `isControlFlowGatingUse`（L452-L466）：判断 use 是否是控制流的"门控"操作数（for 的 lb/ub/step、parallel 的边界、if 的条件）。门控 use + 体内有本核内容 → 该控制流活跃（条件在本核求值）。
- 最后的 fallthrough：**user 不匹配但产生结果** → 结果可能继续流到本核消费者，把 result 全部入 worklist 继续找。

#### 3.5.5 findScopeRelevantUser：worklist 主循环（L520-L547）

```cpp
static Operation *findScopeRelevantUser(Value startValue, StringRef scopeType,
                                        bool ignoreTerminators) {
  SmallVector<Value> worklist{startValue};
  llvm::SmallPtrSet<void *, kSeenValuesCapacity> seenValues;   // 防环

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !seenValues.insert(value.getAsOpaquePointer()).second)
      continue;                                   // 已访问过 → 跳过

    for (OpOperand &use : value.getUses()) {
      if (Operation *found = classifyScopeRelevantUse(
              use, scopeType, ignoreTerminators, worklist)) {
        return found;                             // 找到 relevant 用户
      }
    }
  }
  return nullptr;                                 // 传递链走完，无人使用
}

static Operation *findLiveUser(Value value, StringRef scopeType) {
  return findScopeRelevantUser(value, scopeType, false);   // 不忽略 terminator
}
static Operation *findNonTermUser(Value value, StringRef scopeType) {
  return findScopeRelevantUser(value, scopeType, true);    // 忽略 terminator
}
```

- **算法本质**：对 value 的 use-def 链做了一次**图搜索**（DFS，worklist 实现）。节点是 value，边是"use → user 的 result"。终止条件是找到一个"scope relevant"的用户（属于本核 / 活跃控制流 / 活跃循环携带）。
- `seenValues` 防止环（循环携带依赖会形成 value 的环）。
- 这就是为什么嵌套循环、多级 yield 转发也能正确判断"死活"——旧版只看一层，会把"yield → 循环 result（没人用）"误判为活跃。

### 3.6 【本次更新】循环携带槽位活跃性：needsLoopCarryPreserve（L549-L692）

```cpp
static bool slotHasActiveUse(Value startValue, Operation *owner,
                             unsigned slotIndex, StringRef scopeType) {
  SmallVector<Value> worklist{startValue};
  llvm::SmallPtrSet<void *, kSeenValuesCapacity> seenValues;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !seenValues.insert(value.getAsOpaquePointer()).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      // 三个 checker 依次尝试：condition / yield / 一般 use
      UseCheckResult result;
      if ((result = checkConditionUse(use, owner, slotIndex, scopeType)) != Skip ||
          (result = checkYieldUse(use, owner, slotIndex, scopeType, worklist)) != Skip ||
          (result = checkGeneralUse(use, scopeType, worklist)) != Skip) {
        if (result == UseCheckResult::Active) return true;
      }
    }
  }
  return false;
}

static bool needsLoopCarryPreserve(Operation *owner, unsigned slotIndex,
                                   StringRef scopeType) {
  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    Block &body = forOp.getRegion().front();
    unsigned iterArgIndex = slotIndex + 1;   // body arg 0 是 induction var
    if (iterArgIndex >= body.getNumArguments()) return false;
    return slotHasActiveUse(body.getArgument(iterArgIndex), owner, slotIndex, scopeType);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {   // 新增：while 支持
    Block &before = whileOp.getBefore().front();       // before 块的 arg 直接对应
    Block &after = whileOp.getAfter().front();         // after 块的 arg 也对应
    if (slotIndex < before.getNumArguments() &&
        slotHasActiveUse(before.getArgument(slotIndex), owner, slotIndex, scopeType))
      return true;
    if (slotIndex < after.getNumArguments() &&
        slotHasActiveUse(after.getArgument(slotIndex), owner, slotIndex, scopeType))
      return true;
  }
  return false;
}
```

- **语义**：循环第 `slotIndex` 个携带值，在循环体内（for 的 iterArg / while 的 before+after 块参数）是否被本核真正消费。
- **while 的特殊性**：有 before/after 两个块，两边的 block argument 都对应同一个 result 槽位，任一侧活跃都算保留。
- `checkYieldUse` 里的关键细节：`yieldOwner == owner && idx == slotIndex` 时返回 `Continue`（自己 yield 回自己，不算外部活跃）；同 owner 的**其他**槽位 yield 则把那个 result 入 worklist 继续追。

### 3.7 归一化：neutralizeRegionTerminators（L694-L771，重构）

替代旧版 `neutralizeYieldInRegion`，统一处理 for/if/while 的所有 region terminator：

```cpp
static LogicalResult
neutralizeCarriedTerminatorOperand(Operation *op, const CoreTypeInfo &info,
                                   StringRef scopeType, Location loc,
                                   unsigned slotIndex, OpOperand &operand) {
  if (info.getResultType(slotIndex) == scopeType) return success();  // 属于本核

  if (needsLoopCarryPreserve(op, slotIndex, scopeType))              // 循环内还有人读
    return success();

  Value oldOperand = operand.get();
  // 关键（新增）：循环的携带值如果由"外核"生产，不因父结果有 user 而保留
  bool isLoopOp = isa<scf::ForOp, scf::WhileOp>(op);
  if ((!isLoopOp || !isProducedByForeignScope(oldOperand, scopeType)) &&
      slotIndex < op->getNumResults()) {
    if (Operation *resultUser = findLiveUser(op->getResult(slotIndex), scopeType)) {
      return success();     // 父 op 的这个 result 在本核还有活跃用户 → 保留
    }
  }

  OpBuilder builder(operand.getOwner());
  Value replacement = buildNeutralValue(builder, oldOperand, loc, scopeType);
  if (!replacement) return failure();
  operand.set(replacement);   // 中性值替换
  return success();
}

static LogicalResult neutralizeRegionTerminators(Operation *op, ...) {
  for (Region &region : op->getRegions())
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      unsigned carriedOperandOffset = 0;
      if (isa<scf::ConditionOp>(terminator)) {
        carriedOperandOffset = 1;   // while condition：operand 0 是条件，携带从 1 起
      } else if (!isa<scf::YieldOp>(terminator)) {
        continue;                   // 其他 terminator 不处理
      }
      unsigned numCarriedOperands = terminator->getNumOperands() - carriedOperandOffset;
      for (unsigned i = 0; i < numCarriedOperands; ++i)
        neutralizeCarriedTerminatorOperand(op, info, scopeType, loc, i,
                                           terminator->getOpOperand(i + carriedOperandOffset));
    }
}
```

- **三重保留判断**（都通过才中性化）：
  1. 该槽位属于本核？→ 保留；
  2. 循环体内有本核消费者（`needsLoopCarryPreserve`）？→ 保留；
  3. 父 op 的对应 result 在本核还有活跃用户（`findLiveUser` 传递查找）？→ 保留；但**循环 op + 值由外核生产**（`isProducedByForeignScope`）时跳过第 3 条——外核生产的值不能仅凭"父结果有 user"而在本核苟活。
- **`isProducedByForeignScope`（L684-L692）**：producer 有 `kCoreType` 标记且**没有任何** result 属于本核。

### 3.8 归一化：neutralizeTerminatorUses（L773-L810）

对非 region op 的 terminator 使用进行中性化：

```cpp
static LogicalResult neutralizeTerminatorUses(Operation *op,
                                              const CoreTypeInfo &info,
                                              StringRef scopeType) {
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    if (info.getResultType(i) == scopeType) continue;   // 属于本核，跳过
    Value result = op->getResult(i);
    if (Operation *extraUser = findNonTermUser(result, scopeType)) continue;
    // 收集所有 terminator 使用，替换成中性值
    SmallVector<OpOperand *> usesToNeutralize;
    for (OpOperand &use : llvm::make_early_inc_range(result.getUses()))
      if (use.getOwner()->hasTrait<OpTrait::IsTerminator>())
        usesToNeutralize.push_back(&use);
    for (OpOperand *use : usesToNeutralize) {
      Value replacement = buildNeutralValue(...);
      use->set(replacement);
    }
  }
  return success();
}
```

- `findNonTermUser` 传入 `ignoreTerminators = true`：只有非 terminator 的活跃用户才算"真的在用"。

### 3.9 normalizeRegionOp：region op 归一化（L824-L877）

```cpp
static LogicalResult normalizeRegionOp(Operation *op, StringRef scopeType) {
  auto infoOpt = parseCoreTypeInfo(op);
  if (!infoOpt) { /* 无标记 → 只递归处理嵌套 region */ }

  // 关键（新增）：循环体内没有本核内容，但结果仍被本核消费 → 整个循环保留
  if (isa<scf::ForOp, scf::WhileOp>(op) &&
      !controlFlowOpHasScopeContent(op, scopeType)) {
    if (Operation *resultUser = findLiveResultUser(op, scopeType)) {
      logDebug("preserving complete loop ...");
      return success();     // 不做任何中性化，原样保留
    }
  }

  if (op->getNumRegions() > 0) {
    neutralizeRegionTerminators(op, info, scopeType, loc);   // 先归一化 terminator
    for (Region &region : op->getRegions()) {                // 再递归处理嵌套动作
      auto nestedActions = collectActionsInRegion(region, scopeType);
      executeActions(nestedActions, scopeType);
    }
  }
  return success();
}
```

- **"保留完整循环"分支的场景**：循环体内全是另一个核的计算，但循环的某个 result（比如循环次数计数器）在本核还要用。此时中性化携带值会改变这个 result 的语义 → 只能整个循环保留。
- `regionHasScopeContent` / `controlFlowOpHasScopeContent`（L898-L924 / L441-L448）：递归检查 region 内是否存在属于本核的非 terminator op。

### 3.10 executeActions：执行动作（L945-L989）

```cpp
static LogicalResult executeActions(SmallVector<PendingAction> &actions,
                                    StringRef scopeType) {
  // 逆序执行（从后往前），避免删除 op 影响前面的迭代
  for (auto it = actions.rbegin(); it != actions.rend(); ++it) {
    Operation *op = it->op;
    if (!op) continue;    // 新增：已被间接删除的 op 跳过

    switch (it->kind) {
    case PendingActionKind::EraseDirectly:
      if (!hasLiveUsers(op)) op->erase();      // 无存活用户才删除
      break;
    case PendingActionKind::NormalizeControlFlow:
      if (op->getNumRegions() > 0) normalizeRegionOp(op, scopeType);
      else normalizeNonRegionOp(op, scopeType);
      // 归一化后若变成"死壳"，删除
      if (op && op->getNumRegions() > 0 && isNormalizedDeadShell(op, scopeType)) {
        if (hasLiveUsers(op)) { /* 保留结构 */ }
        else op->erase();
      }
      break;
    }
  }
  return success();
}
```

- 逆序执行是因为删除了前面的 op 会使后面的迭代失效。
- `isNormalizedDeadShell`（L926-L943）：结果无人用 + region 内无本核内容 → 死壳。

### 3.11 separateScopes：单函数拆分主流程（L1000-L1027）

```cpp
static LogicalResult separateScopes(func::FuncOp funcOp) {
  auto scopes = createTwoFullScopes(funcOp);   // 创建 VECTOR + CUBE 两个 scope
  auto vecScope = scopes->first;
  auto cubeScope = scopes->second;

  // 分别收集并执行动作
  auto vecActions = collectActionsInRegion(vecScope.getRegion(), "VECTOR");
  auto cubeActions = collectActionsInRegion(cubeScope.getRegion(), "CUBE");
  if (failed(executeActions(vecActions, "VECTOR")) ||
      failed(executeActions(cubeActions, "CUBE"))) return failure();

  cleanupSsbufferAttrs(funcOp);   // 清理 ssbuffer.core_type 属性
  return success();
}
```

### 3.12 【本次更新】Mechanism A：replaceRedundantVectorStoreLoad（L1029-L1090）

```cpp
// Mechanism A: in VECTOR scopes, replace each redundant store→load pair that
// shares an ssbuffer.transfer_id with the stored value, then erase the load.
//
// The scope-clone step duplicates the CUBE-side scalar load into the VECTOR
// scope too; there it reads back a value this scope itself just stored, so the
// load is pure redundancy (not dead code — it has users), and DCE will not
// remove it.
static void replaceRedundantVectorStoreLoad(scope::ScopeOp scopeOp) {
  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(...);
  if (!coreTypeAttr || coreTypeAttr.getTcoretype() != hivm::TCoreType::VECTOR) {
    return;                                       // 只处理 VECTOR scope
  }

  // Pass 1：收集每个 transfer_id 存进去的值
  llvm::DenseMap<int64_t, mlir::Value> storedValues;
  scopeOp.walk([&](memref::StoreOp storeOp) {
    auto transferIdAttr = storeOp->getAttrOfType<mlir::IntegerAttr>(CVPipeline::kTransferId);
    if (!transferIdAttr) return;
    storedValues[transferIdAttr.getInt()] = storeOp.getValue();
  });
  if (storedValues.empty()) return;

  // Pass 2：同 transfer_id 的 load 直接替换成存储值
  llvm::SmallVector<memref::LoadOp> deadLoads;
  scopeOp.walk([&](memref::LoadOp loadOp) {
    auto transferIdAttr = loadOp->getAttrOfType<mlir::IntegerAttr>(CVPipeline::kTransferId);
    if (!transferIdAttr) return;
    auto it = storedValues.find(transferIdAttr.getInt());
    if (it == storedValues.end()) return;
    mlir::Value storeVal = it->second;
    if (storeVal == loadOp.getResult()) return;

    // 清掉 load 结果上的 annotation.mark volatile——
    // 这个注解只在 CUBE scope 的 load 存活时才有意义
    for (Operation *user : llvm::make_early_inc_range(loadOp.getResult().getUsers())) {
      if (isa<annotation::MarkOp>(user) && user->hasAttr(kMemrefExtVolatile)) {
        user->erase();
      }
    }
    loadOp.replaceAllUsesWith(storeVal);   // load 结果 → 直接用 store 的值
    deadLoads.push_back(loadOp);
  });
  for (memref::LoadOp loadOp : deadLoads) loadOp->erase();
}
```

**问题背景**：`InterCoreTransferAndSync` 处理标量跨核传递时会插 `store + load` 对（配同一个 `transfer_id`）。`createTwoFullScopes` 无脑克隆两份后，VECTOR scope 里也有一份"store 后紧接 load 回来"——**这是纯冗余，但不是死代码**（load 的结果有人用），所以普通 DCE 删不掉，必须显式配对清理。

**做法**：
1. 只针对 VECTOR scope（CUBE 侧的 load 是真正的跨核读取，必须保留）；
2. 两遍扫描：第一遍建 `transfer_id → 存储值` 的表，第二遍找同 id 的 load 做 `replaceAllUsesWith`；
3. 顺带清掉 load 结果上的 `annotation.mark volatile`——volatile 语义是说"这个读取有副作用别删"，但替换后 load 没了，注解悬空还会误导后续 pass。

### 3.13 runOnOperation（L1100-L1133）

```cpp
void mlir::triton::SeparateCVScopePass::runOnOperation() {
  auto module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;   // fallback 兜底

  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp funcOp) { funcOps.push_back(funcOp); });

  for (auto funcOp : funcOps) {
    if (funcOp.getBody().empty()) continue;
    if (failed(separateScopes(funcOp))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  // 给所有 scope 打上 matmul_limited_in_cube 标记
  module.walk([](scope::ScopeOp scopeOp) {
    scopeOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr, UnitAttr::get(...));
  });

  // Mechanism A：清理 VECTOR 侧冗余 store→load 对（新增）
  module.walk([](scope::ScopeOp scopeOp) { replaceRedundantVectorStoreLoad(scopeOp); });
}
```

---

## 四、整体算法总结

```
对每个 function：
1. 把整个 body 克隆成两份 scope（VECTOR + CUBE）
2. 对每个 scope 收集动作：
   - 不属于本核的 op → EraseDirectly（无存活用户才删）
   - 部分匹配/控制流 → NormalizeControlFlow（中性化 yield/condition/结果）
3. 逆序执行动作（删除 + 归一化 + 死壳清理）
4. 清理 ssbuffer.core_type 属性
5. Mechanism A：VECTOR scope 内按 transfer_id 配对，消除冗余 store→load

"是否中性化"的判定链（以循环携带值为例）：
  槽位属于本核？ ── 是 → 保留
       │否
  循环体内有本核消费者？（needsLoopCarryPreserve，worklist 传递查）
       │否
  是循环 op 且值由外核生产？ ── 是 → 直接中性化
       │否
  父 result 在本核还有活跃用户？（findLiveUser，worklist 传递查）
       │否 → 中性化
```

---

## 五、面试要点

1. **为什么要克隆两份再裁剪，而不是直接分类？**
   因为"裁剪"时需要用中性值填充被移除 op 留下的"洞"，直接分类会破坏 SSA use-def 链。先克隆再裁剪，可以让每个 scope 内部保持完整的 SSA 结构（用中性值填充即可）。

2. **"中性化"（neutralize）解决什么问题？**
   当一个 value 在本 scope 里"本不该存在"（属于另一个核），但它的 terminator（yield/return）还在引用它时，会导致 SSA 悬空。用中性值（0 或空 memref）替换，既保持结构合法，又保证该 scope 的实际计算结果不会影响另一核。

3. **为什么删除要"逆序"？**
   IR 修改时，删除前面的 op 会使后面的 op 迭代失效。逆序遍历（从后往前）可以安全地边删边遍历。

4. **worklist 传递查找解决什么旧版 bug？**
   旧版 `findLiveUser` 只看直接 user。一个 value 可能经过 `yield → 循环 result → 再 yield` 的纯转发链后才死亡。只看一层会把"转发中"误判为"有人用"，导致该中性化的值没被中性化、该删的 op 没删（scope 里留下一堆死代码）。新版沿 use-def 图 DFS（worklist + seen 集合防环），把转发链走完才下结论。

5. **为什么 while 和 for 的携带参数处理不一样？**
   for 的 operand 0/1/2 是 lb/ub/step，init 从 operand 3 开始，body arg 0 是 induction variable；while 没有 induction variable，operand 0 起就是携带值，且 before/after 两个块的 arg 都对应 result 槽位。condition terminator 的 operand 0 是条件本身，携带值从 operand 1 开始——归一化时要用 `carriedOperandOffset` 错开。

6. **Mechanism A 清理的 store→load 为什么 DCE 删不掉？**
   DCE 的判据是"结果无人使用"。这个 load 的结果**有人用**（后面的计算消费它），它不是死代码，只是**冗余**（读回了本 scope 刚写进去的值）。必须靠 `transfer_id` 显式配对 store 和 load，用 `replaceAllUsesWith` 把 load 替换成 store 的值，load 才会变成死代码被删。这也解释了为什么 volatile 注解要一起清——它是"别删这个 load"的指令，留着会阻止清理。

7. **这个 pass 之后 IR 长什么样？**
   每个 function 内部变成两个并列的 `scope.scope`（VECTOR 和 CUBE），各自只包含属于自己的计算，通过之前插入的传输缓冲（fixpipe/copy 的目标 memref）与另一核通信。
