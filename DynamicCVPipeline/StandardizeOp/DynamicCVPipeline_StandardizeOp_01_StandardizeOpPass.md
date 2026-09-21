# DynamicCVPipeline 子 Pass 逐行讲解：StandardizeOpPass

> 源码文件：
> - 主 Pass：[`third_party/ascend/lib/DynamicCVPipeline/StandardizeOp.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp.cpp)
> - 子 Pass：[`third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.cpp)
> - Pattern：[`StandardizeOp/FoldExpandExtCollapse.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/FoldExpandExtCollapse.cpp)、[`StandardizeOp/SplitMatmulPattern.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/SplitMatmulPattern.cpp)
>
> 流水线位置：`AddDynamicCVPipelinePass::addPasses` 中 **第 2 个 pass**（`PreCheckAvailable` 之后、`PlanComputeBlock` 之前）

---

## 一、这个 Pass 在做什么

在真正做 CV 拆分之前，把 IR 中 `linalg.matmul` 的形态**标准化**，使其成为"可在 CUBE 上独立计算 + 可在 VECTOR 上补齐累加"的形态：

1. **Matmul 拆分（SplitMatmulPattern，核心）**：`a * b + c` 原本写成一个带累加器（bias）的 matmul；拆成 `(a * b) + c` 两步——matmul 只算乘积（累加器用全零张量），加法单独放一个 `arith.addf/addi` op。这样 matmul 的累加器 `L0C` 完全由 CUBE 自己维护，不需要从 VECTOR 侧传入，后续拆分才能干净地把 matmul 划给 CUBE、把加法划给 VECTOR。
2. **Shape 归一化（FoldExpandExtCollapse）**：把 `collapse(ext(expand(x)))` 这种"整形 + 位宽扩展"的组合折叠成直接的 `ext(x)`，减少后续分析要穿越的中间 op。
3. **适用性探测**：拆分过程中如果发现"matmul 可能不会执行"（例如外层 for 的 upper ≤ lower、if 条件非真），给模块打上 `ERRCODE_IGNORED` fallback——kernel 不适合做动态 CV 流水线，回退标准编译。

**为什么放在第 2 步**：后续所有 Pass（PlanComputeBlock 分块、SplitDataflow 拆分、传输插入）都假设 matmul 已经处于"乘积 + 加法"的干净形态；先标准化能避免每个下游 Pass 重复处理带累加器的 matmul。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `linalg::MatmulOp` | Linalg 方言的矩阵乘，DPS 风格：`DpsInputs`（A、B）+ `DpsInits`（累加器 bias） |
| `L0C` | CUBE 核的累加器缓存。matmul 结果直接落在 L0C 上 |
| `L1` | CUBE 核的共享缓存。另一个 matmul 把结果当 A/B 输入时需经 L1 |
| L0C→L0C cascade | matmul 结果作为下一个 matmul 的 **bias**（累加器）。在 NPUIR 后端若跨 block 会出现 fixpipe 错误 |
| L0C→L1 | matmul 结果作为下一个 matmul 的 **A/B 输入**。跨 block 时需要落地全局内存 |
| `kAddFromMatmul` | `"ssbuffer.add_from_matmul"`，标记拆分后补的加法 op，供 `DataDependencyAnalysis` 识别 |
| `kLoopCarriedL0C` | `"ssbuffer.loop_carried_l0c"`，标记已被拆分处理的 matmul，防止 greedy rewrite 重复处理 |
| `kMayNotExec` | `"ssbuffer.may_not_exec"`，标记"可能不执行"的 matmul（无法用 select 保护的场景） |
| `kForMayNotExec` | `"ssbuffer.for_may_not_exec"`，标记可能不执行 for 下补出的 select/fill |
| `kHIVMMatmulLimitedInCubeAttr` | `"hivm.matmul_limited_in_cube"`，matmul 相关计算已收敛在 CUBE 内的提示 |
| `kMatmulAtLeastOnceHint` | `"matmul_at_least_once"` 注解：证明 matmul 至少执行一次，可关闭 mayNotExec 保护 |
| `needSplitAllFuncNme` | 硬编码的函数名黑名单（当前为空字符串表），命中则无条件拆分所有 matmul |
| `top-down` greedy | `GreedyRewriteConfig().setUseTopDownTraversal()`，跨 for 块的 matmul 依赖要求自上而下匹配 |

---

## 三、逐行讲解

### 3.1 主 Pass：`StandardizeOp.cpp`（L45-L83）

```cpp
void StandardizeOpPass::runOnOperation() {
  auto op = getOperation();
  if (CVPipeline::hasFallbackAttr(op)) { return; }         // fallback 兜底

  OpPassManager pm(op.getOperationName());
  pm.addPass(createPatternMatchRewritePass());             // 唯一子 Pass
  if (llvm::failed(runPipeline(pm, op))) {                 // 子 pipeline 失败
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // 拆分后检查：是否存在"可能不执行"的 matmul
  bool findMayNotExec = false;
  op->walk([&](linalg::MatmulOp matmulOp) {
    if (matmulOp->hasAttr(CVPipeline::kMayNotExec)) { findMayNotExec = true; }
  });
  if (findMayNotExec) {
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_IGNORED);  // 回退标准编译
    return;
  }
}
```

- 主 Pass 是一个**编排壳**：真正的变换在子 Pass `PatternMatchRewritePass` 里。
- 子 pipeline 失败（rewrite 不收敛）→ `ERRCODE_FAILED`；
- 拆分后若仍有 `kMayNotExec` matmul → `ERRCODE_IGNORED`。区别在于：`FAILED` 是"我们出错了"，`IGNORED` 是"这个 kernel 本就不适合 CV 流水线"，两者都会让 `AddDynamicCVPipeline` 走回退，但日志语义不同。
- 注册（L76-L83）：`createStandardizeOpPass()` + `registerStandardizeOpPasses()`（同时注册子 Pass `createPatternMatchRewritePass`）。

### 3.2 子 Pass：`PatternMatchRewritePass`（PatternMatchRewrites.cpp L44-L78）

```cpp
void PatternMatchRewritePass::runOnOperation() {
  auto moduleOp = getOperation();
  if (CVPipeline::hasFallbackAttr(moduleOp)) { return; }

  bool needSplitAll = false;
  moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(needSplitAllFuncNme, funcOp.getSymName()))
      return WalkResult::advance();                 // 不在名单 → 继续找
    needSplitAll = true;                            // 命中名单 → 全部拆
    return WalkResult::interrupt();
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SplitMatmulPattern>(ctx, needSplitAll);
  patterns.add<FoldExpandExtCollapse>(ctx);

  // 跨 for 块的 matmul 依赖要求 pattern 自上而下匹配
  GreedyRewriteConfig config = GreedyRewriteConfig().setUseTopDownTraversal();
  if (llvm::failed(applyPatternsGreedily(moduleOp, std::move(patterns), config))) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

- `needSplitAllFuncNme[]{""}`（L42）：当前是空表，正常情况恒为 `false`，是预留的调试开关。
- 两个 pattern 一起进 greedy driver，`applyPatternsGreedily` 反复应用直到 fixpoint。
- **为什么 top-down**：`SplitMatmulPattern` 要处理"外层 for 可能不执行"的场景，且跨 for 块的 cascade matmul 依赖需要先处理外层再处理内层；top-down 保证匹配顺序符合依赖方向。

### 3.3 Pattern：`FoldExpandExtCollapse`（FoldExpandExtCollapse.cpp L42-L96）

匹配形状：`collapse( extU/SI( expand(x) ) )`，折叠为 `extU/SI(x, resultType)`。

```cpp
LogicalResult FoldExpandExtCollapse::matchAndRewrite(
    tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const {
  auto collapseReassoc = collapseOp.getReassociationIndices();
  // 只支持把前两维 [0,1] 合并成二维的 collapse
  if (collapseReassoc.size() != 1 || collapseReassoc[0].size() != 2 ||
      collapseReassoc[0][0] != 0 || collapseReassoc[0][1] != 1)
    return failure();

  auto extOp = collapseOp.getSrc().getDefiningOp();       // 必须是 ext 类
  auto extUIOp = dyn_cast<arith::ExtUIOp>(extOp);
  auto extSIOp = dyn_cast<arith::ExtSIOp>(extOp);
  if (!extUIOp && !extSIOp) return failure();

  if (!extOp->hasOneUse()) return failure();              // ext 只被 collapse 用
  auto expandOp = extOp->getOperand(0).getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandOp || !expandOp->hasOneUse()) return failure();
  // expand 也必须是 [0,1] 前两维展开
  if (expandReassoc.size() != 1 || ...) return failure();

  Value extInput = expandOp.getSrc();
  Type resultType = collapseOp.getResult().getType();
  // collapse(ext(expand(x))) == ext(x)：整形两两抵消，位宽扩展保留
  if (extUIOp) {
    rewriter.create<arith::ExtUIOp>(collapseOp.getLoc(), resultType, extInput);
  } else {
    rewriter.create<arith::ExtSIOp>(...);
  }
  rewriter.replaceOp(collapseOp, newExt.getResult());
  return success();
}
```

- 只处理前两维的 `[0,1]` 重组：这对应 triton 中常见的 `(M,N)` 二维张量经 reshape 变成三维再做元素级扩展的场景。
- `collapse`（把三维压回二维）与 `expand`（把二维撑成三维）形状上互为逆操作，中间的 `ext`（位宽扩展）保持不变——于是整体等价于直接在原输入上做 `ext`。这个折叠消除了两个 shape op，让后续的 use-def 链更短。

### 3.4 Pattern：`SplitMatmulPattern`（SplitMatmulPattern.cpp，核心，L1-L1003）

#### 3.4.1 数据结构与工具（L70-L135）

```cpp
struct MatmulInputs { Value a; Value b; Value bias; };   // bias = DpsInits[0]
struct SplitInfo {
  bool mayNotExec;     // matmul 外层可能存在 0 次执行的路径
  Value outerInValue;  // 沿 for/if 链追溯到最外层初始值（原 bias 的源头）
  Value outerOutValue; // 沿 for/if 链追溯到最外层结果
  bool shouldSplit;    // 是否执行拆分
  bool supported = false; // mayNotExec 是否可用 select 保护
};
```

- `parseMatmulInputs`（L90-L95）：`{inputs[0], inputs[1], inits[0]}`。
- `operationIsFillZero`（L98-L106）：bias 是 `linalg.fill(0)` → 无需拆分（加零无意义）。
- `isFloatOrInt`（L109-L112）：元素类型必须是 float/int。
- `scfMayNotExec`（L114-L129）：判断控制流可能不执行：
  - `scf.for`：`upper ≤ lower`（常量已知时直接比较，未知时保守返回 `true`）；
  - `scf.if`：条件不是常量 `true` 就认为可能不执行。

#### 3.4.2 `searchInArgsChain`：向上追溯外层链（L138-L227）

从 matmul 结果出发，沿 `for/if` 逐层外推，得到**最外层结果** `outerOutValue` 和**最外层初始值** `outerInValue`，同时回答 `argsLimitedInMatmul`（bias 链上是否只被 matmul 使用）。

对 `scf.for`：
- 当前 block argument 必须是 for 的迭代参数（`argIdx = argNumber - 1`）；
- 该参数在块内的所有用户必须只回到本 matmul 或其链，且本 matmul 结果只被 yield 用（且 yield 槽位一致）；
- 满足则 `outerInValue = forOp.getInitArgs()[argIdx]`，`nextSearchValue = forOp->getResult(argIdx)`，继续外推。

对 `scf.if`：
- 要求有 else 块；本分支 yield 与另一分支 yield 在同一个槽位分别对应 `nextSearchValue` 与 `outerInValue`；
- 满足则 `nextSearchValue = ifOp.getResult(resultIdx)`，继续外推。

每层外推后：`mayNotExec |= scfMayNotExec(parentOp)`。

**结论**：如果 `argsLimitedInMatmul == false`（bias 链上有非 matmul 的旁路使用，S25 场景），说明 bias 的价值不只在 matmul 里，不能把 bias 从 matmul 中拆出，直接返回"应该拆分但把外层原样保留"。

#### 3.4.3 `traceChainUser`：沿用户链追踪（L249-L319）

通用工具：给定 value，沿 uses 追踪匹配 op。

```cpp
std::optional<Operation *> traceChainUser(Value value, bool needSingle,
    const std::function<bool(Operation*, Value)> &isMatchedOp,
    const std::function<bool(Operation*, Value)> &isSkipOp) {
  // 1. 先扫描一次：统计本块内的用户；needSingle 时多于一个 → nullopt
  // 2. 对每个 use：
  //    - ViewLikeOpInterface / ExtractSliceOp → 穿透其 result 继续
  //    - scf::ForOp（作为 init，恰好 1 次）→ 穿透到 regionIterArgs[initIndx] 继续
  //    - scf::YieldOp → 映射回父 op 的对应 result 继续
  //    - isMatchedOp 命中 → 返回该 op
  //    - isSkipOp 命中（单结果）→ 穿透其 result 继续
}
```

- 它封装了"跨 `view`、跨 `for` init、跨 `yield`"的 SSA 穿透逻辑，所有"结果被谁消费"的查询都复用它。
- `isSkipOp` 用来跳过不关心的中间 op（如 `linalg.transpose`）。

#### 3.4.4 L0C cascade 与全局内存判断（L321-L428）

- `hasCrossBlockCascadeL0CConsumer`（L321-L356）：matmul 结果沿链被另一个 matmul 当作 **bias** 消费（`matchMatmulC` 要求 `a != value && b != value && bias == value`），且消费者在**不同 block** 的 for 中（同一 block 内是合法 cascade）→ 返回 true，同时给当前 for 打 `kMayNotExecNPU`。这是避免 NPUIR fixpipe 错误的保守标记。
- `hasCrossBlockCascadeL0CProducer`（L363-L387）：bias 定义来自**另一个 block** 的 matmul（跨块 L0C→L0C producer）→ 需要拆分。
- `isSubviewFromGlobalMemory`（L389-L428）：沿 `viewSource` 逐层向上，若最终是 `func` 参数（全局内存）返回 true；若是 `for/while` 的 block argument，继续溯源到 init（`for` 用 `argIndex-1`，`while` 用 `argIndex`）；任何非 view 的中间 op → false。

#### 3.4.5 输入/输出过滤决策（L447-L581）

`isOutputFilter`（L447-L491）两个场景直接判定拆分：
1. **store 到全局内存**且跨 block：结果被 `bufferization::MaterializeInDestinationOp` 或 `hivm::StoreOp` 消费，dest 溯源到全局内存（`isSubviewFromGlobalMemory`）→ 拆（避免 NPUIR 插入 fixpipe 报错）；
2. **单用户是 L0C bias** 且跨 block（`traceChainUser(..., needSingle=true)`）→ 拆。

`shouldSplitByOutput`（L492-L517）：除上述外，结果被另一 matmul 当 **A/B 输入**（`matchMatmulAB`，可穿透 `linalg.transpose`）且跨 block → 拆（L0C→L1 需要落地全局内存，S01-S08）。

`isInputFilter`（L533-L548）：bias 来自跨 block 的 matmul（L0C producer）→ 拆。

`shouldSplitByInput`（L549-L581）：
1. `isInputFilter` 命中 → 拆；
2. bias 无定义 op → 拆；
3. bias 是 `fill(0)` → **不拆**；
4. bias 是广播且 BT size ≤ `CACHE_TABLE_BUFFER_SIZE`(4096)（`getBTSizeFromValidBroadcastOp`）→ **不拆**（小的广播 bias 直接进 cache table，无需拆分）；
5. bias 来自 matmul（L0C remain）→ **不拆**；
6. 其余 → 拆。

#### 3.4.6 决策链：`verifyMatmul` + `shouldSplit`（L590-L711）

`verifyMatmul`（L590-L612）：已带 `kLoopCarriedL0C`（已处理过）、无 inits / inputs<2（非法）、bias 非 `RankedTensorType`（非 tensor 模式）、元素非 float/int → 都不拆。

`shouldSplit`（L670-L711）完整决策：
1. `verifyMatmul` 失败 → `nullopt`（不匹配，pattern 放弃）；
2. `needSplitAll` → 无条件拆；
3. `searchInArgsChain` 得到 `argsLimitedInMatmul / mayNotExec / outerInValue / outerOutValue`；若结果带 `kMatmulAtLeastOnceHint` 注解，则 `mayNotExec = false`（有证据至少执行一次）；
4. `!argsLimitedInMatmul` → 拆（S25）；
5. `mayNotExec` → 走 `handleMayNotExec` 特判；
6. `!shouldSplitByInput && !shouldSplitByOutput` → 不拆；
7. 否则 → 拆。

#### 3.4.7 mayNotExec 特判：`handleMayNotExec`（L616-L648）

只支持最常见场景：matmul 直接位于可能不执行的 `scf.for` 下，且 bias 是该 for 的迭代参数。
- bias 对应 init 为 `fill(0)` 时：若结果有跨块 cascade L0C 消费者 → `{mayNotExec=false, shouldSplit=false, supported=false}`（保持原样，标记留给下游）；否则返回 `{mayNotExec=true, supported=true}`（可拆 + 用 select 保护）；
- 若 init 由跨块 cascade L0C producer 产生 → `{mayNotExec=true, shouldSplit=false, supported=true}`（不拆但打 `kMayNotExec` 标记）。
- 其余返回默认 `{mayNotExec=true, shouldSplit=true, supported=false}`。

#### 3.4.8 变换执行：`splitMatmul`（L820-L952）

把 `matmul(a,b,bias)` 拆成 `add(new_matmul(a,b,zero), bias)`：

1. **构造零累加器**（L835-L868）：
   - 在 `outerOutValue` 定义点插入 `tensor.empty` + `fill(0)` 得到 `zeroVal`；
   - 若 `outerDefOp == matmul`：把 matmul 自己的 DpsInit[0]（bias）替换成 `zeroVal`；
   - 否则：把 `outerInValue` 的、被 `outerDefOp` 使用的部分替换成 `zeroVal`。
2. **创建新 matmul**（L870-L893，非 mayNotExec 或没有外层 for 时）：
   - 在原 matmul 位置创建 `matmul(a, b, bias)`（累加器仍是原 bias，但此时 bias 已经被替换逻辑保证是零填充链上的值）；
   - 复制原 matmul 属性，剔除 `operandSegmentSizes / res_attrs / arg_attrs`；
   - `replaceOp(matmulOp, newMatmul)`。
   - 若 `outerOutValue == matmul.getResult(0)`，更新 `outerOutValue = newMatmul.getResult(0)`。
3. **补加法**（L895-L951）：
   - 在 `outerOutValue` 之后插入 `arith.addf/addi(newOutValue, outerInValue)`；
   - mayNotExec 时：先插入 `cmp(ub>lb)` + `fill(0)` + `select`，把 `newOutValue` 换成 select 结果，select/fill 打 `kForMayNotExec`，for 打 `kHIVMMatmulLimitedInCubeAttr`；
   - 加法 op 打 `kAddFromMatmul`；
   - `outerOutValue.replaceUsesWithIf(addOp_result, 排除 preservedUsers)`。

#### 3.4.9 `matchAndRewrite` 入口（L954-L1003）

1. fallback 检查 → failure；
2. `shouldSplit` 得到 `SplitInfo`；无结果 → failure；
3. 给 matmul 打 `kLoopCarriedL0C`（防重复）；
4. 若 `outerOutValue` 带 `kMatmulAtLeastOnceHint` 注解：移除注解，注解 op 空壳时 erase，父 op 打 `kHIVMMatmulLimitedInCubeAttr`；
5. `shouldSplit` → `splitMatmul`；
6. 否则若 `mayNotExec`：
   - `supported` → `handleMayNotExecSelect`（插入 select 保护）；
   - 否则 → matmul 打 `kMayNotExec`（触发主 Pass 的 `ERRCODE_IGNORED` 回退）。

---

## 四、算法流程总结

```
StandardizeOpPass
├─ PatternMatchRewritePass（top-down greedy）
│   ├─ SplitMatmulPattern：为每个 matmul 决策 + 拆分
│   │   决策链：verifyMatmul → searchInArgsChain(外推 for/if)
│   │          → argsLimitedInMatmul? → mayNotExec? → 输入/输出过滤
│   │   变换：matmul(a,b,bias) → add(new_matmul(a,b,zero), bias)
│   │         mayNotExec → 插入 cmp + select 保护 / 打 kMayNotExec
│   └─ FoldExpandExtCollapse：collapse(ext(expand(x))) → ext(x)
└─ walk 检查：存在 kMayNotExec matmul → ERRCODE_IGNORED 回退
```

复杂度：`applyPatternsGreedily` 迭代至 fixpoint；每个 matmul 的决策含 use-def 链追溯，最坏 O(use 链长)；嵌套 for 的 cascade 场景会逐层外推。

---

## 五、面试要点

1. **StandardizeOp 到底"标准化"了什么？**
   核心是把 `a*b+c` 从"带累加器的 matmul"重写成"纯乘积 matmul + 独立加法"。动机：matmul 的累加器 L0C 属于 CUBE，若 bias 来自 VECTOR 侧计算，跨核拆分时累加器无法干净归属；拆出加法后，matmul 完全归 CUBE、加法归 VECTOR，SplitDataflow 的 `core_type` 标注才干净。

2. **为什么拆分决策里有"bias 是 fill(0) 就不拆"？**
   加零无意义。若 bias 是零填充，`a*b+0 == a*b`，拆分只会增加一个多余的加法 op；直接保留原 matmul 更优。同理广播且小的 bias（≤4096B）走 cache table 也无需拆。

3. **`mayNotExec` 是什么？怎么处理？**
   外层 for 的 `upper ≤ lower` 或 if 条件可能为假时，matmul 可能不执行，此时拆分后的加法会引入"未定义初值"问题。处理分三档：支持场景（matmul 直接在可能不执行的 for 下）插入 `cmp(ub>lb) ? matmul结果 : 0` 的 select 保护（打 `kForMayNotExec`）；有 `kMatmulAtLeastOnceHint` 证明至少执行一次则关掉保护；其余场景 matmul 打 `kMayNotExec`，主 Pass 发现后整体回退 `ERRCODE_IGNORED`——宁可不用 CV 流水线，也不生成错误语义。

4. **L0C→L0C / L0C→L1 跨 block 为什么必须拆？**
   NPU 硬件上 matmul 结果在 L0C，若被另一 block 的 matmul 当 bias（L0C→L0C）用，NPUIR 后端插入 fixpipe 会报错；若当 A/B 输入（L0C→L1）用，需要落地全局内存再搬回。这两种跨 block 数据流都必须通过拆分 + 后续 SplitDataflow 插入显式传输来解决。同一 block 内（同一 for 体内）的 cascade 是允许的，因此判断都要求"跨 block"。

5. **`searchInArgsChain` 在做什么？和 `traceChainUser` 什么区别？**
   `searchInArgsChain` 向上（producer 方向）追溯 for/if 链，找最外层 init/result，回答"bias 链是否只属于 matmul"；`traceChainUser` 向下（consumer 方向）沿 uses 找匹配用户，回答"结果被谁消费"。两者配合完成拆分决策的两端分析。

6. **`kAddFromMatmul` 属性的作用？**
   标记拆分后补出的加法 op。`DataDependencyAnalysis` 在分析依赖时，遇到 `kAddFromMatmul` 会把它当作 matmul 的附属处理（见 04 篇 `removeVectorPseudoOps`：VECTOR 侧这类加零伪 op 会被清理），保证后续拆分结果正确。

7. **为什么要"先标准化"再分块？**
   PlanComputeBlock 按 op 粒度分块、SplitDataflow 按 `core_type` 切分，都基于 matmul 形态稳定。先在此 pass 统一 matmul 结构，下游无需再处理"带累加器的 matmul"这一特例，降低每个 Pass 的复杂度。

8. **主 Pass 为什么是编排壳？**
   标准 MLIR 实践：一个逻辑阶段由"小 Pass 组合"实现，便于单独调试、复用、控制 pipeline 粒度。`StandardizeOpPass` 只负责创建子 pipeline 并汇总结果（失败码/适用性探测）。
