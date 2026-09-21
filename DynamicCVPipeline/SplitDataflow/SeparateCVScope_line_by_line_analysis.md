# SeparateCVScope.cpp 逐段逐行分析

> 分析对象：[`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/SeparateCVScope.cpp`](third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/SeparateCVScope.cpp)
>
> 本文严格按照源码从上到下讲解。这里的“逐行”以连续语义段为单位：相邻且共同完成一个动作的代码放在同一小节说明，避免把函数签名、花括号和每个条件机械拆开后丢失语义。

## 1. 文件的最终目的

这个 Pass 读取每个操作上的 `ssbuffer.core_type`，把一个函数的完整操作流复制成两份：

- 一份放入 `VECTOR` 类型的 `scope::ScopeOp`；
- 一份放入 `CUBE` 类型的 `scope::ScopeOp`。

随后分别删掉或中和不属于当前 scope 的部分。对于普通数据流，删除无用操作即可；对于 `scf.for`、`scf.while` 等控制流，不能简单删除，因为循环参数、循环结果、`scf.yield` 和 `scf.condition` 在位置上必须对应。因此代码还需要判断某个 loop-carried slot 是否仍被当前 scope 使用，并在确定无用时用零值或临时内存替换。

---

## 2. 文件头、依赖和调试设施（第 1～77 行）

### 2.1 许可证（第 1～21 行）

文件开头是 MIT 风格许可证，规定代码可使用、复制、修改和分发，同时声明不提供担保。

### 2.2 头文件（第 23～47 行）

- `<optional>`：`parseCoreTypeInfo()` 用 `std::optional` 表达属性不存在或解析失败。
- `<map>`：Mechanism A（`replaceRedundantVectorStoreLoad`）用 `DenseMap` 之外也可容纳的映射结构；本文件实际用 `llvm::DenseMap`，`<map>` 是遗留或防御性 include。
- `SmallPtrSet`、`SmallVector`：用于小规模 worklist、访问集合和动作列表，减少小对象场景下的堆分配。
- `llvm/Support/Debug.h`：提供 `LLVM_DEBUG` 和 `llvm::dbgs()`。
- `Utils.h`：提供 CV Pipeline 的属性名、fallback 标记、`kTransferId` 等公共能力。
- `SSBufferManager.h`：标量同步与 `ssbuffer.transfer_id` 标记的来源约定（Mechanism A 依赖该 transfer_id 语义）。
- HIVM 和 Scope dialect：创建 VECTOR/CUBE scope，并设置核心类型属性。
- Arith、Func、MemRef、SCF：创建零常量、内存分配，以及识别和处理结构化控制流。
- `IRMapping`：克隆第一个完整 scope 时建立 SSA 映射。
- `LoopLikeInterface`：统一访问 `scf.for`、`scf.while` 一类循环的初始化参数和结果。
- `RegionUtils.h`：Region 层面的辅助遍历/工具（与 pass 基础设施配套）。
- Pass 相关头文件：实现并注册 MLIR Pass。

### 2.3 命名空间、调试宏和常量（第 49～73 行）

- `using namespace mlir` 和 `using namespace mlir::triton` 简化类型名。
- `DEBUG_TYPE` 把调试输出归入 `SeparateCVScope` 分类。
- `DBGS()` 生成带统一前缀的调试流。
- `logDebug()` 是可变参数模板，仅在启用 LLVM debug 时把所有参数顺序写入日志。
- `kForOpOperandPrefixCount = 3` 表示 `scf.for` 的前三个 operand 是 lower bound、upper bound、step，后面才是 iter operands。
- `kSeenValuesCapacity = 16` 是两个 SSA 遍历器的内联访问集合容量。
- `debugDumpOperation()` 在调试模式下打印操作变换前后的完整 IR。

### 2.4 前置声明（第 75～77 行）

提前声明：

- `needsLoopCarryPreserve()`：判断循环某个 carried slot 是否必须保留；
- `regionHasScopeContent()`：递归判断 region 内是否存在当前 scope 的有效内容。

它们会被前面定义的函数调用，而实际实现位于文件后部，因此需要前置声明。

---

## 3. `ssbuffer.core_type` 的解析和匹配（第 79～153 行）

### 3.1 `CoreTypeInfo`（第 79～109 行）

`resultTypes` 保存按结果位置拆分后的核心类型。例如属性值 `"VECTOR,CUBE"` 表示结果 0 属于 VECTOR，结果 1 属于 CUBE。

#### `getResultType(index)`

- 索引合法时返回对应元素；
- 索引越界时返回第一个元素。

后者兼容只写一个核心类型、但操作有多个结果的情况，即把第一个类型视作默认类型。该函数假定 `resultTypes` 非空，这由解析函数保证。

#### `hasResultForScope(scopeType)`

遍历所有结果类型，只要存在一个等于目标 scope 就返回 `true`。它回答“这个操作是否至少部分属于当前 scope”。

#### `allResultsMatchScope(scopeType, numResults)`

- 零结果操作使用第一个属性值判断自身所属 scope；
- 有结果操作逐槽调用 `getResultType()`；
- 所有结果都属于当前 scope 才返回 `true`。

这个函数用于区分“整个操作可原样保留”和“混合结果操作需要归一化”。

### 3.2 `parseCoreTypeInfo()`（第 111～132 行）

执行步骤：

1. 从操作读取字符串属性 `ssbuffer.core_type`；没有属性就返回 `nullopt`。
2. 读取原始字符串，用逗号分割，且不保留空项。
3. 如果没有得到任何部分，返回 `nullopt`。
4. 对每一项执行 `trim()`，去掉首尾空白。
5. 把结果写入 `CoreTypeInfo::resultTypes` 并返回。

这里没有拒绝 trim 后的空字符串；因此格式合法性主要依赖更早的分析阶段。

### 3.3 `ScopeMatchInfo`、`getScopeMatchInfo()` 和 `matchesScope()`（第 134～153 行）

`ScopeMatchInfo` 同时保存两个维度：

- `matches`：至少有一个结果属于目标 scope；
- `allResultsMatch`：所有结果都属于目标 scope。

`getScopeMatchInfo()`：

1. 解析属性；无属性时返回两个 `false`。
2. 零结果操作根据属性第一项判断。
3. 有结果操作通过 `hasResultForScope()` 判断部分匹配。
4. 再通过 `allResultsMatchScope()` 判断完全匹配。

`matchesScope()` 是只关心第一个维度时的简化包装。

---

## 4. 构造中性替代值（第 155～181 行）

`buildNeutralValue()` 为准备断开的 SSA 边构造同类型占位值：

1. 输入 Value 为空，直接返回空 Value。
2. 若类型是 `MemRefType`，创建 `memref::AllocOp`，返回新分配内存。
3. 若是其他 `ShapedType`：
   - 仅支持静态 shape；
   - 获取元素类型零属性；
   - 构造全零 `DenseElementsAttr`；
   - 创建 `arith::ConstantOp`。
4. 若是标量等普通类型，尝试创建对应零常量。
5. 无法构造时返回空 Value，由调用者把它转换成 Pass failure。

参数 `scopeType` 当前未被函数体使用，只保留在接口中，可能是为日志、策略扩展或历史实现预留。

“中性”只表示类型和结构合法，不保证数学语义上的单位元。因为这些值理论上应位于当前 scope 的死路径/死槽中，其主要用途是维持 IR 结构合法。

---

## 5. 复制完整函数为两个 scope（第 183～232 行）

`createTwoFullScopes()` 是结构变换的起点。

1. 检查函数 body 只有一个 block；否则发出错误并失败。
2. 取得最后也是唯一的 block。
3. 收集 terminator 之前的全部操作。
4. 若函数内没有待移动操作，返回两个空的 `scope::ScopeOp` 句柄。
5. 在原最后一个操作之后建立 builder。
6. 创建第一个 `scope::ScopeOp`，并显式给 body region 添加 block。
7. 将原函数操作逐个 `remove()` 后插入该 scope，而不是克隆；这份成为 VECTOR 原件。
8. 在 scope body 末尾创建 `scope::ReturnOp`。
9. 把插入点移到 VECTOR scope 后，使用 `IRMapping` 克隆整个 scope；克隆体成为 CUBE scope。
10. 创建 HIVM VECTOR/CUBE 枚举属性并分别附着到两个 scope。
11. 返回二者。

结果是两个 scope 在初始时完全相同，后续再分别裁剪。

---

## 6. 操作分类和待执行动作收集（第 234～313 行）

### 6.1 `isPureForScope()`（第 234～259 行）

判断一个带 region 的操作能否在当前 scope 原样保留：

1. 操作本身必须匹配当前 scope，且所有结果都匹配。
2. 遍历所有 region、block 和嵌套操作。
3. terminator 跳过，因为它通常没有独立的 core type。
4. 嵌套操作还有 region 时递归检查。
5. 叶子操作必须匹配当前 scope。

任何一处不满足就返回 `false`。这是一个偏严格的“整个子树纯净”判断。

### 6.2 `PendingActionKind` 与 `PendingAction`（第 261～266 行）

收集阶段不立即修改 IR，而是记录：

- `EraseDirectly`：目标操作原则上属于另一 scope，可直接删除；
- `NormalizeControlFlow`：操作包含混合 scope 数据或 region，必须先处理中间 SSA 和 terminator。

`PendingAction` 保存操作指针和动作类型。

### 6.3 `isControlFlowOp()`（第 268～270 行）

当前认可的结构化控制流包括：

- `scf.for`；
- `scf.if`；
- `scf.while`；
- `scf.parallel`。

### 6.4 `collectActionForOp()`（第 272～280 行）

用于无 region 的普通操作：

- 完全不匹配当前 scope：记录直接删除；
- 部分匹配但并非所有结果匹配：记录归一化；
- 所有结果匹配：不记录动作，原样保留。

### 6.5 `collectActionsInRegion()`（第 282～303 行）

从一个 region 的当前层收集动作：

1. 遍历 block 和操作，跳过 terminator。
2. 对带 region 的操作：
   - 若属于受支持控制流，并且不是当前 scope 的纯子树，则记录归一化；
   - 若不是受支持控制流且自身不匹配，则记录直接删除；
   - 然后 `continue`，此时不直接下钻，嵌套内容留到归一化阶段处理。
3. 对叶子操作调用 `collectActionForOp()`。

这体现“先收集、后修改”，避免遍历过程中删除操作导致迭代器失效。

### 6.6 `hasLiveUsers()`（第 305～313 行）

只要操作任一结果仍存在 owner 有效的 use，就认为有结构性用户。它不判断用户是否属于当前 scope，是删除前的保守安全检查。

---

## 7. 第一套活性分析：从结果向外寻找当前 scope 用户（第 315～547 行）

这套分析回答：从某个 SSA Value 出发，沿数据流继续传播，最终是否到达当前 scope 必须保留的操作？

### 7.1 `hasActiveNestedLoopCarryUse()`（第 318～339 行）

判断某个 use 是否正好进入循环的 init/carried 参数：

1. 将 use owner 转为 `LoopLikeOpInterface`；不是循环则返回 `false`。
2. 获取循环 init operands；为空则返回 `false`。
3. 根据 operand 编号计算它是否落在 init 区间。
4. 把 init operand 相对位置换算成循环 result slot。
5. 调用 `needsLoopCarryPreserve()` 检查这个槽在循环体内部是否被当前 scope 活跃使用。

因此，“值只是进入循环”并不自动算活跃；还必须证明对应 carried slot 真正影响当前 scope。

### 7.2 `canSkipForwardingUse()`（第 342～377 行）

该函数识别“仅把值转发到非当前 scope 循环结果”的 use，可在外部活性搜索中忽略。

对 `scf.for`：

1. 前三个 operand 是边界和步长，不能按 carried slot 跳过。
2. 其余 operand 映射到 result slot。
3. 只有该结果槽不属于当前 scope，且 `needsLoopCarryPreserve()` 证明循环内也不需要时，才返回 `true`。

对 `scf.while`：

1. operand 编号直接作为 slot 编号，越界不跳过。
2. 当前 slot 属于目标 scope 时不跳过。
3. loop-carried slot 内部活跃时不跳过。
4. 最后还要求 while 的所有直接用户都不匹配当前 scope，才允许跳过。

它是活性分析中的“剪枝”，不是 IR 修改。

### 7.3 `shouldPreserveFromYield()`（第 380～405 行）

当值进入 `scf.yield` 时，判断是否应把循环 owner 本身当作活跃用户：

1. 先调用 `needsLoopCarryPreserve()`；循环内无活跃证据则不保留。
2. owner 不是 `LoopLikeOpInterface` 或没有 core type 时，直接采用上述判断。
3. 对有 core type 的循环，还要求存在另一个属于当前 scope 的 sibling result。

这个额外条件主要面向混合 scope 多结果循环，避免因为单个外域槽的纯回传而过度保留整个循环。

### 7.4 `acceptUser()`（第 408～416 行）

统一处理“找到相关用户”后的返回规则：

- 用户为空或已经脱离 block，返回空；
- `ignoreTerminators == false` 时允许返回 terminator；
- `ignoreTerminators == true` 时过滤 terminator。

### 7.5 `handleYieldUse()`（第 420～438 行）

处理值被 `scf.yield` 使用的情况：

1. 取得 yield 的父操作和 operand slot。
2. 若 slot 对应父操作结果：
   - `shouldPreserveFromYield()` 为真，返回父操作作为活跃用户；
   - 否则把父操作对应 result 放入 worklist，继续向循环外追踪。
3. 若没有对应结果，但父操作属于当前 scope 或属于控制流，则把父操作作为相关用户。
4. 否则返回空。

这一步建立了“region terminator operand → parent operation result”的跨 region SSA 传播。

### 7.6 `controlFlowOpHasScopeContent()`（第 441～449 行）

遍历控制流操作的每个 region，调用 `regionHasScopeContent()`。任何 region 内含当前 scope 操作即返回 `true`。

### 7.7 `isControlFlowGatingUse()`（第 452～467 行）

识别决定控制流是否执行的输入：

- `scf.for` 的 lower/upper/step；
- `scf.parallel` 每个维度的 lower/upper/step，共 `3 * numLoops` 个；
- `scf.if` 的 operand 0 条件。

若控制流内部有当前 scope 内容，这些 gating value 就必须视作活跃。

### 7.8 `classifyScopeRelevantUse()`（第 470～518 行）

这是第一套分析的核心决策链，顺序很重要：

1. 无效用户、用户已脱离 block，或 `canSkipForwardingUse()` 命中：忽略。
2. use 进入活跃 nested loop-carried slot：接受该用户。
3. use owner 是 `scf.condition`：
   - parent 本身匹配 scope，则接受 parent；
   - operand 0 是 while 条件且 parent 内有 scope 内容，也接受 parent；
   - 其他 condition 参数暂不视为活跃。
4. use owner 是 `scf.yield`：交给 `handleYieldUse()`。
5. 普通用户直接匹配当前 scope：接受。
6. 用户没有结果：
   - 控制流操作仍视为活跃；
   - 其他不匹配操作无法继续传播，忽略。
7. use 是控制流 gating 输入且控制流内部含当前 scope 内容：接受用户。
8. 其他有结果操作：把它的全部结果加入 worklist，跨过中间转发节点继续搜索。

这里的语义是“找到一个证据即可返回”，不是枚举所有活跃用户。

### 7.9 `findScopeRelevantUser()`（第 520～539 行）

实现 worklist 图遍历：

1. 从 `startValue` 初始化栈式 worklist。
2. 用 `SmallPtrSet` 保存访问过的 Value opaque pointer，避免 SSA 环路或重复传播。
3. 取出一个 Value，空值或已访问则跳过。
4. 遍历它的所有 uses，调用 `classifyScopeRelevantUse()`。
5. 一旦找到相关 Operation 立即返回。
6. worklist 耗尽仍无证据则返回空。

### 7.10 `findLiveUser()` 与 `findNonTermUser()`（第 541～547 行）

二者复用同一引擎：

- `findLiveUser()` 不忽略 terminator；
- `findNonTermUser()` 忽略 terminator，用于判断除了结构回传以外是否还有真实消费者。

---

## 8. 第二套活性分析：循环 carried slot 内部是否活跃（第 549～681 行）

第一套从“某个结果”向外追踪；第二套从“循环 block argument”向循环体内追踪。目标是回答特定 slot 能否安全中和。

### 8.1 `UseCheckResult`（第 549 行）

- `Skip`：当前检查器不负责这种 use，请交给下一个检查器；
- `Active`：发现当前 scope 的活跃使用；
- `Continue`：此 use 已处理，但尚未发现活跃证据，继续 worklist。

### 8.2 `checkConditionUse()`（第 551～572 行）

只处理 `scf.condition`：

1. 非 condition 返回 `Skip`。
2. 若不是当前 owner 对应 slot 的标准 condition carried operand：
   - 如果是某个 condition 的 operand 0，且其 parent 内有当前 scope 内容，则为 `Active`；
   - parent 不匹配当前 scope时，把相关传播交给后续过程，返回 `Continue`；
   - 其他情况保守视作 `Active`。
3. 若正是当前 owner 的 `slotIndex + 1` carried operand，则返回 `Continue`，避免纯回边本身被误判为有效消费。

`+1` 是因为 `scf.condition` 的 operand 0 是布尔条件。

### 8.3 `checkYieldUse()`（第 574～601 行）

只处理 `scf.yield`：

- 若 yield 属于当前 owner：
  - 对其他 slot 的交叉传播，把对应 owner result 加入 worklist；
  - 当前 slot 的直接回传不算活跃；
  - 返回 `Continue`。
- 若 yield 属于嵌套 owner，且 operand 对应其结果，则把嵌套 owner result 加入 worklist。
- 若无法映射结果，但嵌套 owner 匹配 scope 或是控制流，则返回 `Active`。
- 其他情况继续。

### 8.4 `checkGeneralUse()`（第 603～620 行）

处理非 condition、非 yield 的一般 use：

1. 进入嵌套循环活跃 carried slot，返回 `Active`。
2. 用户无结果时：匹配当前 scope或属于控制流即 `Active`，否则 `Continue`。
3. 用户有结果时，把所有结果加入 worklist，继续沿 SSA 数据流传播。

### 8.5 `slotHasActiveUse()`（第 622～652 行）

它和第一套分析一样使用 worklist 与访问集合，但对每个 use 依次尝试三个专用检查器：

1. `checkConditionUse()`；
2. 若返回 `Skip`，再调用 `checkYieldUse()`；
3. 若仍为 `Skip`，再调用 `checkGeneralUse()`。

只要某个检查器返回 `Active` 就立即返回 `true`；遍历结束无证据则返回 `false`。

### 8.6 `needsLoopCarryPreserve()`（第 654～681 行）

对不同循环建立“结果槽 → region block argument”的映射：

- `scf.for`：block argument 0 是 induction variable，因此 result slot `i` 对应 body argument `i + 1`；从该参数调用 `slotHasActiveUse()`。
- `scf.while`：同一个 slot 可能在 before region 和 after region 中被使用，因此分别检查两个 region 的第 `i` 个 block argument；任一活跃即保留。
- 其他 owner 返回 `false`。

这是整份代码判断循环参数是否可被中和的核心入口。

---

## 9. 中和控制流和 terminator 数据边（第 683～810 行）

### 9.1 `isProducedByForeignScope()`（第 684～692 行）

检查 operand 的 defining op：

1. 没有 producer，或 producer 没有 core type 属性，不认定为外域生产者。
2. producer 完全没有当前 scope 的结果时，认定为 foreign scope producer。

这允许后续逻辑断开“另一个 scope 的 producer → 当前循环 terminator”的边，避免仅因父循环 result 的用户而错误保留外域 producer。

### 9.2 `neutralizeCarriedTerminatorOperand()`（第 694～734 行）

逐个处理某个控制流操作的 carried slot：

1. slot 本身属于当前 scope：不处理。
2. `needsLoopCarryPreserve()` 证明循环体内需要：不处理。
3. 保存 terminator 当前 operand。
4. 若父操作不是 for/while，或者旧 operand 不是外域 producer，则继续检查父操作同 slot result 是否仍有当前 scope 活跃用户：有则不处理。
5. 在 terminator 前创建与旧 operand 同类型的中性值。
6. 构造失败则记录日志并返回 failure。
7. 用新值替换 terminator operand。

这个函数有三层保护：slot 类型匹配、循环内部活性、父结果外部活性。

### 9.3 `neutralizeRegionTerminators()`（第 736～771 行）

遍历操作所有 region 和 block：

1. 空 region 跳过。
2. 只处理 `scf.condition` 和 `scf.yield` terminator。
3. condition 的 carried operands 从位置 1 开始；yield 从位置 0 开始。
4. 对每个 carried operand 调用上一函数。
5. 任一替换失败则整体失败。

### 9.4 `neutralizeTerminatorUses()`（第 773～810 行）

用于没有 region、但具有混合 scope 多结果的普通操作：

1. 遍历每个结果槽，当前 scope 槽跳过。
2. 对外域结果调用 `findNonTermUser()`；若仍有非 terminator 活跃用户则保留。
3. 收集这个结果的所有 terminator uses。使用 `make_early_inc_range` 防止后续改 use 时破坏遍历。
4. 为每个 terminator use 构造同类型中性值并替换。
5. 构造失败则返回 failure。

它不会改普通用户，只切断结构化回传边。

---

## 10. 归一化操作（第 812～896 行）

### 10.1 `executeActions()` 前置声明（第 812～813 行）

`normalizeRegionOp()` 会递归调用 `executeActions()`，而后者定义在更后面，因此先声明。

### 10.2 `findLiveResultUser()`（第 815～822 行）

`normalizeRegionOp` 的前置辅助函数：遍历操作的每个 result，对每个 result 调用 `findLiveUser()`；任何一个 result 有当前 scope 活跃用户就返回该用户。它把“循环结果是否仍被消费”的判定抽成单一入口，供“完整循环提前保留”使用。

### 10.3 `normalizeRegionOp()`（第 824～877 行）

处理带 region 的控制流或容器操作。

#### 情况一：操作没有 `ssbuffer.core_type`

1. 不直接报错，因为容器/控制流本身可能没有属性。
2. 对每个 region 收集嵌套动作。
3. 执行嵌套动作。
4. 完成后返回 success。

#### 情况二：操作有 core type

1. 保存解析结果和 location，打印归一化前 IR。
2. 对 `scf.for`/`scf.while` 做完整循环保护：
   - 如果循环体完全没有当前 scope 内容；
   - 但任一循环结果通过 `findLiveResultUser()`（内部调用 `findLiveUser()`）仍被当前 scope 消费；
   - 则原样保留整个循环并提前返回。

这个提前返回避免中和 carried values 后改变仍被消费的循环结果。

3. 若操作有 region，先调用 `neutralizeRegionTerminators()` 中和不需要的 carried slots。
4. 再逐 region 收集并执行嵌套动作，从内层删除不匹配操作。
5. 打印归一化后 IR并返回。

注意顺序是“先修 terminator 边，再改嵌套操作”。此时后续删除 producer 时更不容易留下悬空依赖。

### 10.4 `normalizeNonRegionOp()`（第 879～896 行）

1. 必须存在 core type；缺失则发出 IR error 并失败。
2. 打印变换前 IR。
3. 调用 `neutralizeTerminatorUses()`。
4. 成功后打印变换后 IR。

---

## 11. region 内容、死壳判断和动作执行（第 891～982 行）

### 11.1 `regionHasScopeContent()`（第 891～917 行）

递归搜索 region：

- 跳过 terminator；
- 带 region 的非受支持控制流操作，如果自身匹配则立即返回 true；
- 对所有子 region 继续递归；
- 叶子操作匹配当前 scope则返回 true；
- 全部遍历结束无匹配则返回 false。

对受支持控制流，重点看内部内容而不是只看控制流 owner 的属性。

### 11.2 `isNormalizedDeadShell()`（第 919～936 行）

判断归一化后的 region 操作是否只剩结构外壳：

1. 操作必须存在且有 region。
2. 任一结果还有当前 scope 活跃用户，则不是死壳。
3. 任一 region 内还有当前 scope 内容，也不是死壳。
4. 两者都没有才返回 true。

### 11.3 `executeActions()`（第 938～982 行）

动作按反向顺序执行。这样通常先处理后出现的 consumer，再处理前面的 producer，降低删除依赖链时的冲突。

#### `EraseDirectly`

只有 `hasLiveUsers()` 为 false 才调用 `erase()`。仍有 SSA 用户时保守保留。

#### `NormalizeControlFlow`

1. 有 region 调用 `normalizeRegionOp()`；无 region 调用 `normalizeNonRegionOp()`。
2. 失败立即向上传播。
3. 对带 region 的操作调用 `isNormalizedDeadShell()`。
4. 若是死壳但仍有结构性用户，则记录日志并保留。
5. 若是死壳且没有任何用户，则删除整个操作。

这里把“语义上对当前 scope 无内容”和“MLIR 结构上可安全 erase”分开判断。

---

## 12. 清理属性和单函数主流程（第 984～1020 行）

### 12.1 `removeSsbufferAttrs()` 与 `cleanupSsbufferAttrs()`（第 984～991 行）

- 前者从单个操作删除 `ssbuffer.core_type`；
- 后者先清理根操作，再 `walk()` 清理所有后代。

拆分完成后该临时分析属性不再需要。

### 12.2 `separateScopes()`（第 993～1020 行）

单函数处理流程：

1. 打印变换前函数。
2. 调用 `createTwoFullScopes()` 创建两份完整 IR。
3. 创建失败则返回 failure。
4. 分别在 VECTOR 和 CUBE scope 收集动作。
5. 分别执行动作；任一失败则整个函数失败。
6. 清理函数及后代的 `ssbuffer.core_type`。
7. 打印变换后函数并返回 success。

VECTOR 和 CUBE 的裁剪逻辑完全复用，只通过 `scopeType` 参数区分。

### 12.3 Mechanism A：消除 VECTOR 侧冗余的 scalar store→load 对（第 1029～1090 行）

`replaceRedundantVectorStoreLoad()` 是 Pass 末尾新增的冗余优化，只对 `VECTOR` 类型 scope 生效：

1. 读取 scope 的 `hivm::TCoreTypeAttr`，不是 VECTOR 就直接返回。
2. **Pass 1（收集）**：`walk()` 所有 `memref::StoreOp`，把带 `ssbuffer.transfer_id` 的 store 记录成 `transfer_id → 被存储的值` 映射；没有任何记录则直接返回。
3. **Pass 2（替换）**：`walk()` 所有 `memref::LoadOp`，查找同 `transfer_id` 的 store 值：
   - 找不到记录，或 `storeVal == loadOp.getResult()`（自环），跳过；
   - 先删除附在 load 结果上的 `annotation::MarkOp`（带 `kMemrefExtVolatile` 的 volatile 注解），因为该注解只在 CUBE scope 的 scalar load 存活时有意义；
   - 用 store 值 `replaceAllUsesWith` 替换 load，并把它收集进 `deadLoads`。
4. 最后统一 `erase()` 所有 deadLoad，避免 walk 过程中删除操作。

**为什么需要它**：`createTwoFullScopes()` 克隆完整函数时，CUBE 侧从共享 buffer 读回标量值的 scalar load 也被复制进了 VECTOR scope。在 VECTOR scope 里，这个 load 读的是本 scope 自己刚 store 进去的值，属于纯冗余——但它有用户，不是死代码，DCE 无法删除。只有依赖 `ssbuffer.transfer_id` 的配对信息才能识别并消除。

---

## 13. Pass 接口、执行入口和注册（第 1092～1146 行）

### 13.1 `getDependentDialects()`（第 1092～1098 行）

声明 Pass 可能创建或使用：

- Annotation dialect（Mechanism A 检查和删除 volatile 注解）；
- Arith dialect；
- HIVM dialect；
- MemRef dialect；
- Scope dialect。

SCF 和 Func 主要被读取和改写，相关上下文通常已加载；这里列的是 Pass 明确依赖创建的 dialect。

### 13.2 `runOnOperation()`（第 1100～1133 行）

模块级执行步骤：

1. 获取当前 `ModuleOp`。
2. 模块已有 fallback 属性时直接退出，避免在上游失败后继续变换。
3. 先收集所有 `func::FuncOp`，避免 walk 过程中改 IR 影响遍历。
4. 打印 Pass 前模块。
5. 逐函数处理：
   - 空函数体跳过；
   - `separateScopes()` 失败时给模块设置 `ERRCODE_FAILED` fallback 属性并停止。
6. 所有函数成功后，遍历每个 `scope::ScopeOp`，设置 `kHIVMMatmulLimitedInCubeAttr` 单位属性。
7. Mechanism A：再 walk 每个 `scope::ScopeOp`，调用 `replaceRedundantVectorStoreLoad()` 清理 VECTOR 侧冗余 store→load。
8. 打印 Pass 后模块。

### 13.3 创建与注册（第 1135～1146 行）

- 回到 `mlir::triton` 命名空间。
- `createSeparateCVScopePass()` 构造 `SeparateCVScopePass`。
- `registerSeparateCVScopePasses()` 用工厂 lambda 注册 Pass。

---

## 14. 按源码顺序总结

整份文件从上到下可以压缩为以下过程：

1. 解析每个操作的 scope 归属。
2. 为断开的 SSA 边准备同类型占位值。
3. 把单 block 函数复制为 VECTOR/CUBE 两份完整 scope。
4. 只读遍历每份 scope，收集删除或归一化动作。
5. 通过第一套活性分析寻找 scope 相关的外部用户。
6. 通过第二套活性分析判断 loop-carried slot 的循环内部用途。
7. 在安全条件满足时中和 `yield`/`condition` 的外域 carried operand。
8. 递归删除各层不属于当前 scope 的操作。
9. 删除无内容、无用户的控制流死壳。
10. 清理临时属性，设置后续 Pipeline 所需 scope 属性。
11. 对 VECTOR scope 消除冗余的 scalar store→load 对（Mechanism A），清理克隆产生的、DCE 无法删除的冗余。
