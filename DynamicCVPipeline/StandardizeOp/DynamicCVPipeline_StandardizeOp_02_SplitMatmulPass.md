# DynamicCVPipeline 子 Pass 逐行讲解：SplitMatmulPass

> 源码文件：[`SplitMatmul.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/SplitMatmul.cpp)、[`SplitMatmul.h`](../../../third_party/ascend/include/DynamicCVPipeline/StandardizeOp/SplitMatmul.h)
>
> 依赖 Common 工具：[`ControlFlowGraph.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/Common/ControlFlowGraph.cpp)（452 行，值流图 + SCC）
>
> 流水线位置：`StandardizeOpPass` 的 **第 2 个子 Pass**（`PatternMatchRewritePass` 之后、`PackMemrefPass` 之前）
>
> 相对旧版本：这是 **main-dev 新增的独立 Pass**，取代了旧版"存在 `kMayNotExec` matmul → `ERRCODE_IGNORED` 整体回退"的策略。

---

## 一、这个 Pass 在做什么

处理 **L0C 累加器复用链（cascade matmul 链）的拆分 + may-not-execute 保护**：

1. **识别 L0C 复用链**：基于 `ControlFlowGraph`（按"累加器初值"边构建 C 值流图）做强连通分量（SCC）分析，把互为可达的 matmul 聚成 `L0CReuseUnit`，再沿单 use 边把 unit 链成 producer→consumer 的 `L0CReuseLink`；
2. **链级拆分**：对 `shouldSplit` 的链，把链首输入清零（`createZero`）、链尾输出后补 `arith.addf/addi`（原输入加回），等价于 `chain_result + origin_input`；
3. **MNE guard**：对 "may-not-execute" 的链，给链上每个 matmul 分配一个跨核 SSBuffer 计数器（执行后 `store 1`），链尾插入 `scf.if` 判断"所有 matmul 都没执行则输出零张量"，防止链被完全跳过时输出未定义值。

**为什么需要独立的 Pass**：L0C 累加器复用（一个 matmul 的结果作下一个的 bias）是 NPU 上最复杂的数据流——旧版 per-matmul pattern 无法感知"多个 matmul 串成的 cascade 链"的整体结构，只能逐个决策，容易漏拆分或重复拆分。整链分析才能正确处理链首清零、链尾加回、以及整链可能不执行的保护。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `L0CReuseUnit` | L0C 复用单元：一个 SCC 分量内的 matmul 集合（含 `inputValue`/`outputResult`/`entryOp`/`inputOpOperand`/`isMNE`） |
| `L0CReuseAdjacency` | unit 间邻接：`next`（我的输出是谁的输入）、`prev`（我的输入是谁的输出），单 use 且双向校验 |
| `L0CReuseLink` | L0C 复用链：沿邻接把 unit 串成线性链（producer→consumer），`chain.front()` 为链首、`chain.back()` 为链尾 |
| `ControlFlowGraph` | 值流图：按"是不是累加器初值 `DpsInits[0]`"过滤边，提供 `getStronglyConnectedComponents`（SCC） |
| `SCC` | 强连通分量：值流图中互为可达的 op 闭包；一个 SCC = 一个会"循环复用" L0C 的区域 |
| `shouldSplitMatmul` | 判断链首输入是否需要拆分：`tensor.empty`/`fill(0)`/小 broadcast（≤`CACHE_TABLE_BUFFER_SIZE`）→ 不拆 |
| `mayNotExecute` | 沿父 op 链判断 region 是否可能执行 0 次（for 上界≤下界、while 条件为 0、if 条件非 1） |
| `MNE guard` | 每个 matmul 一个 SSBuffer 计数器 + `scf.if` 判"全没执行则输出零"，防止未定义输出 |
| `kMNEStore0` | `"ssbuffer.mne_store_0"`，标记计数器清零的 `memref.store` |
| `kCoupledMatmulAndStore` | `"ssbuffer.coupled_matmul_and_store"`，成对 id 标记 matmul 与其"执行后 store 1" |
| `kMayNotExecNPU` | `"ssbuffer.may_not_exec_npu"`，标记可能不执行的输出定义点，供下游 NPU 后端识别 |
| `kAddFromMatmul` | `"ssbuffer.add_from_matmul"`，标记链尾补出的加法，供 DataDependencyAnalysis 识别 |

---

## 三、逐行讲解

### 3.1 数据结构（L55-L136）

```cpp
using UnitId = unsigned;
struct L0CReuseUnit {
  SmallVector<linalg::MatmulOp> matmuls; // 本单元内的 matmul（SCC 内可能有多个）
  Value inputValue;                      // 单元累加器输入（初值）
  OpResult outputResult;                 // 单元输出（结果）
  Operation *entryOp = nullptr;          // 链首 entry op（插入清零处）
  OpOperand *inputOpOperand = nullptr;   // 输入的 operand 引用（改写处）
  bool isMNE = false;                    // may-not-execute
};
struct L0CReuseAdjacency { SmallVector<int> next; SmallVector<int> prev; };

struct L0CReuseLink {
  SmallVector<UnitId> units;             // 链上的 unit id（前→后）
  SmallVector<linalg::MatmulOp> matmuls; // 展平后的 matmul 列表
  Value inputValue;                      // 链首输入
  OpResult outputResult;                 // 链尾输出
  bool shouldSplit = true;               // 是否拆分
  bool isMNE = false;                    // 整链是否可能不执行
  ...
  static L0CReuseLink build(...);        // 从 seed 出发 DFS 组装链
};
```

- `L0CReuseLink::build`（L84-L106）：从 seed 出发，先用 `dfsProducer` 沿 `prev` 向上（producer 方向）后序遍历，把 producer 侧按数据流顺序压入；再取 `next[seed]` 用 `dfsConsumer` 沿 `next` 向下（consumer 方向）把后继压入。因邻接描述的是"每个 unit 至多一个 producer/consumer"的**不相交线性链**，DFS 不会遇到已访问节点（L80-L83 注释）。
- `dfsProducer`（L108-L121）：先递归 producer，再 `push_back(now)`——后序保证 producer 在前；`dfsConsumer`（L123-L135）：先 `push_back(now)` 再递归 next——前序保证 consumer 在后。

### 3.2 Pass 类声明（L138-L167）

`SplitMatmulPass` 私有方法（互相配合完成六阶段流水）：
- `collectL0CReuseUnits`：从 CFG 收集 L0C 复用单元；
- `linkReuseL0COps`：把 unit 链成 L0C 复用链；
- `analyzeSplitAndMNE`：逐 unit/链分析 `isMNE` 与 `shouldSplit`；
- `printAnalysis`：debug 输出；
- `splitMatmulChain`：单条链的拆分（清零 + 补 add）；
- `setAttrForMatmulChain`：链上打 `kHIVMMatmulLimitedInCubeAttr` / `kMayNotExecNPU`；
- `splitMatmulChains`：遍历所有链执行拆分；
- `insertMNEGuardUB`：插入 MNE guard。

`getArgument()` 返回 `"ssbuffer-split-matmul"`，`getDescription()` 为 "Split accumulator reuse chains using a control-flow-aware graph"。

### 3.3 `regionMayNotExecute`（L188-L213）

判断一个 region（for/if/while）是否可能执行 0 次：
- `scf.for`：`lb`/`ub` 常量已知且 `lb < ub` 才证明必执行，否则保守返回 `true`（可能不执行）；
- `scf.if`：条件不是常量 `1` → `true`；
- `scf.while`：若 child 在 before 区 → 返回 `false`（before 必执行）；否则看 `scf.condition` 的条件是否为非零常量，条件恒为 0 → `true`；
- 未知 region 语义 → `false`（无法证明，不判为不执行）。

### 3.4 `mayNotExecute`（L215-L247）

unit 级判断：
1. 输出带 `kMatmulAtLeastOnceHint` 注解 → `false`（有证据至少执行一次）；
2. 只有 1 个 matmul 且其 result 就是 outputValue → `false`（单 matmul 无 cascade）；
3. 否则沿每个 matmul 的父 op 链向上走 `regionMayNotExecute`，若**任一 matmul 的所有父链都判定可能不执行**，则整 unit `isMNE=true`。遍历上限到 `scope`（chain 末端输出 op 所在 block），把检查限定在复用作用域内。

### 3.5 `isValidL0CReusedComponent`（L249-L289）

筛选合法的 SCC 分量（否则跳过不处理）：
1. 分量必须恰有 1 个输入 operand、1 个输出 result，且 owner 一致（即 C 值流只进一次、出一次）；
2. 分量内必须含 `linalg::MatmulOp`；
3. **后端限制**：除最终输出 `output` 外的 internal 累加器值只能沿 C/SCF 值流边流转，若存在外部 use 指向非 output 的内部值 → 跳过（后端不支持 internal accumulator observers）。

### 3.6 `collectL0CReuseUnits`（L291-L329）

- 用 `graph.getStronglyConnectedComponents(isMatmul)` 得到所有 SCC；
- 每个合法分量构造一个 `L0CReuseUnit`（把分量内 matmul 全部纳入 `assigned`）；
- 末尾再 walk 一次：未被分量覆盖的 matmul 单独各成一个 unit（`inputValue = DpsInits[0]`、`outputResult = 唯一结果`、`entryOp = mm`）。保证每个 matmul 恰有一个 owner。

### 3.7 `linkReuseL0COps`（L331-L379）

把 unit 链成链：
1. `inputToIndex` / `outputToIndex`：只对**单 use** 的 input/output 建索引（多 use 不参与链接）；
2. 构建邻接：`next[now]` = "我的 output 被某个 unit 当 input"；`prev[now]` = 双向校验（该 unit 确实通过 `next` 指向我）；
3. 从每个未访问的 seed 出发 `L0CReuseLink::build` 组装链。

### 3.8 `shouldSplitMatmul`（L381-L403）

判断链首输入是否需要拆分（返回 `false` 即不拆）：
- 无定义 op 或定义是 `tensor::EmptyOp` → 不拆；
- `linalg::BroadcastOp` 且 BT size ∈ (0, `CACHE_TABLE_BUFFER_SIZE`] → 不拆（小广播进 cache table）；
- `linalg::FillOp` 且填充常量是零（`m_Zero`/`m_AnyZeroFloat`）→ 不拆（加零无意义）；
- 其余 → 拆。

### 3.9 `analyzeSplitAndMNE`（L405-L416）

- 逐 unit 计算 `unit.isMNE = mayNotExecute(unit)`；
- 逐链计算 `link.shouldSplit = shouldSplitMatmul(link.inputValue)`、`link.isMNE = all_of(units 的 isMNE)`（整链可能不执行 ⇔ 链上每个 unit 都可能不执行）。

### 3.10 `splitMatmulChain`（L443-L489）

链级拆分：
1. `createZero`（L443-L457）：用 `tensor.empty` + `arith.constant(0)` + `linalg.fill` 造与输入同型的零张量；
2. 在 `entryOp` 处插入零张量，`inputUse->set(zero)`（链首的累加器初值清零）；
3. 保存原 `outputResult` 的所有 uses；
4. 在链尾 `exit = outputResult.getDefiningOp()` 之后插入 `arith.addf`（浮点）或 `arith.addi`（整型）：`add(chain_output, origin_input)`，打 `kAddFromMatmul`；
5. 把原 uses 全部改指向 `add` 结果；`link.inputValue = zero`（供后续 MNE guard 用）。

### 3.11 `setAttrForMatmulChain`（L491-L508）

- 链上每个 matmul 的 `inner = mm->getParentOp()`：未打 `kHIVMMatmulLimitedInCubeAttr` 就打上（提示 matmul 已收敛在 CUBE 内）；
- 每个 unit 的 `outer = outputResult.getDefiningOp()`：若 `isMNE` 且未打 `kMayNotExecNPU`，打 `kMayNotExecNPU`（供下游 NPU 后端识别可能不执行的输出点）。

### 3.12 `splitMatmulChains`（L510-L519）

遍历所有链，`shouldSplit && inputOpOperand && entryOp` 时调用 `splitMatmulChain`。

### 3.13 `insertMNEGuardUB`（L522-L604）

对 may-not-execute 的链插入 MNE guard：
1. 为链上每个 matmul 分配一个 SSBuffer 计数器（`manager.allocateMNECounterAddr()`），在 `entry` 处 `store 0`（打 `kMNEStore0`），在 matmul 之后 `store 1`（与 matmul 成对打 `kCoupledMatmulAndStore` 的递增 id）；
2. 在链尾输出之后：逐计数器 `load`，`cmp eq 0` 得到"是否跳过"，`and` 合并得到 `allSkipped`；
3. 插入 `scf.if(allSkipped)`：then 分支 yield 零张量（同型 zero fill），else 分支 yield 原 `outputResult`；
4. 把原 uses 全部改指向 `if` 结果。

**语义**：若因控制流导致链上所有 matmul 都没执行（计数器全 0），输出用零张量兜底，避免读到未定义的 L0C 残留值。

### 3.14 `runOnOperation`（L606-L678，六阶段）

```cpp
void SplitMatmulPass::runOnOperation() {
  // 前置：只有存在 kMayNotExec matmul 才工作
  bool findMayNotExec = false;
  module->walk([&](linalg::MatmulOp m) {
    if (m->hasAttr(kMayNotExec)) findMayNotExec = true; });
  if (!findMayNotExec) { return; }              // 无则不处理
  if (hasFallbackAttr(module)) { return; }
  // Stage1: 建 C 值流图 + 收集 L0C 复用单元
  ControlFlowGraph graph(module, /*isAccumulatorInit*/ ...);
  collectL0CReuseUnits(l0cReusUnits, graph);
  // Stage2: 链成 L0C 复用链
  linkReuseL0COps(l0cReusUnits, l0cReuselinks);
  // Stage3: 决策（先分析后改 IR）
  analyzeSplitAndMNE(l0cReusUnits, l0cReuselinks);
  // Stage4: 链级拆分
  if (failed(splitMatmulChains(builder, l0cReuselinks))) { setFallbackAttr(IGNORED); }
  // Stage5: MNE guard
  for (link : l0cReuselinks) if (link.isMNE) insertMNEGuardUB(...);
  // Stage6: 打属性 + 清理 at_least_once 注解
  for (link : l0cReuselinks) setAttrForMatmulChain(...);
  for (unit : l0cReusUnits) cleanup kMatmulAtLeastOnceHint 注解;
}
```

关键点：
- **入口 gate**：`findMayNotExec == false` 直接 return——本 Pass 只为"可能不执行"的场景服务，正常的必执行 matmul 已由 `SplitMatmulPattern` 处理；
- `ControlFlowGraph` 的边过滤 lambda：只有 matmul 的 `DpsInits[0]`（累加器初值）边被纳入，非 matmul op 全纳入——这样图只关心"C 值流"；
- Stage6 清理 `kMatmulAtLeastOnceHint`：注解为空壳（无其余 attrs）时 erase 注解 op。

---

## 四、算法流程总结

```
SplitMatmulPass
├─ 入口 gate: 无 kMayNotExec matmul → return（正常 matmul 已由 Pattern 处理）
├─ Stage1  ControlFlowGraph(C 值流) → SCC → L0CReuseUnit（合法分量筛选）
├─ Stage2  linkReuseL0COps：单 use 邻接 → L0CReuseLink（线性 chain）
├─ Stage3  analyzeSplitAndMNE：unit.isMNE + link.shouldSplit/isMNE
├─ Stage4  splitMatmulChains：shouldSplit → 链首清零 + 链尾补 add
├─ Stage5  insertMNEGuardUB：isMNE 链 → SSBuffer 计数器 + scf.if 零兜底
└─ Stage6  setAttrForMatmulChain(kHIVMMatmulLimitedInCube/kMayNotExecNPU)
           + 清理 kMatmulAtLeastOnceHint 注解
```

复杂度：SCC 分析 O(V+E)；链组装 DFS O(N)；MNE guard 每必保护 matmul O(1)（一个计数器 + store）。

---

## 五、面试要点

1. **SplitMatmulPass 和 SplitMatmulPattern 什么关系？**
   分工互补：`SplitMatmulPattern`（PatternMatchRewritePass 内）是 greedy pattern，逐 matmul 做基础 `a*b+c→add(matmul(a,b,0),c)` 拆分；`SplitMatmulPass` 是独立 Pass，**只处理存在 `kMayNotExec` 标记的场景**，做 L0C 复用链的整链拆分与 may-not-execute 保护。

2. **为什么要基于 SCC / ControlFlowGraph 而不是逐 matmul 分析？**
   L0C 累加器复用（cascade）是"多个 matmul 串成链/环"的数据流，逐 matmul 决策看不到整条链的结构，会漏拆环上的 matmul 或重复拆分。SCC 先把"互为可达"的 matmul 闭包聚成 unit，再沿单 use 边拉成线性链，才能正确做链首清零、链尾加回。

3. **MNE guard 为什么需要 SSBuffer 计数器，而不是直接 `scf.if(条件)`？**
   May-not-execute 由多层 for/while/if 的嵌套条件决定，无法在链尾用单一标量条件表达。用"每 matmul 一个计数器、执行后 store 1、链尾判全 0"的方式，把"链上所有 matmul 都被跳过"这一运行时事实显式记录到共享内存，再用 `scf.if` 读计数器做兜底，无需追溯控制流条件。

4. **为什么链首清零、链尾加回？**
   旧 IR 里 matmul 的累加器初值来自上游（可能是 VECTOR 侧计算或上一个块的结果）。拆分成"纯乘积"后，累加器必须先清零，让 CUBE 独立算 `a*b`；链尾再把原始初值 `+` 回去，恢复 `a*b+c` 语义。这样 matmul 归类 CUBE、加法归类 VECTOR，跨核边界干净。

5. **`shouldSplitMatmul` 的"不拆"条件为什么是 empty/fill(0)/小 broadcast？**
   这些 bias 本质是"零或可缓存的小张量"，拆分只会白增一个加法 op；`a*b+0 == a*b`，小广播 bias 可直接进 cache table。反之，bias 来自真实计算（如上游 matmul 结果）才必须拆，否则 L0C 累加器无法干净归属 CUBE。

6. **入口 gate 为什么是"无 kMayNotExec 就 return"？**
   本 Pass 只补旧 pattern 处理不了的一类场景（可能不执行的 cascade 链）。正常必执行的 matmul 已被 `SplitMatmulPattern` 拆干净，此处无需重复工作；提前 return 也避免无谓的建图、SCC 开销。

7. **`kMayNotExecNPU` 和 `kMayNotExec` 的区别？**
   `kMayNotExec`（`ssbuffer.may_not_exec`）是 Pattern 阶段打的"此 matmul 可能不执行、需要保护"标记；`kMayNotExecNPU`（`ssbuffer.may_not_exec_npu`）是本 Pass 在**链级分析后**给输出定义点打的标记，专门提示 NPU 后端该点可能不执行（供 fixpipe/调度做保守处理）。两者打的位置和下游消费者不同。