# DynamicCVPipeline 子 Pass 逐行讲解：PackMemrefPass

> 源码文件：[`PackMemref.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/PackMemref.cpp)
>
> 依赖 Common 工具：[`CustomOpUtils.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/Common/CustomOpUtils.cpp)（`CustomOpAnalysis`）、[`ScopeOpUtils.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/Common/ScopeOpUtils.cpp)（`packScopeOp`/`unpackScopeOp`）
>
> 流水线位置：`StandardizeOpPass` 的 **第 3 个子 Pass**（`SplitMatmulPass` 之后）；是 **main-dev 新增**的子 Pass。

---

## 一、这个 Pass 在做什么

把每个 `hivm.hir.custom` op **与其本地 buffer 一起打包进一个 `scope::ScopeOp`**，并打上分析所得的核心类型（`ssbuffer.core_type`）。

`hivm.hir.custom`（自定义算子）通常不是孤立存在的——它通过 `memref.alloc → bufferization.to_tensor → custom`（或 `alloc → custom → to_tensor`，当 custom op 填充 buffer 时）的链与若干本地 buffer 绑定。若不提前打包，后续 `OpClassifier` 看到的是一串零散的 buffer 链，难以把它们与 custom op 关联、进而正确归类核型。

打包的产出是一个"自包含的 scope 单元"：内部是 custom op 及其 buffer 的构建链，对外只暴露逃逸的值，并带一个 `kCoreType` 属性作为（可能的）核型提示。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `hivm::CustomOp` | 自定义算子（`hivm.hir.custom`），块级计算的基本单元，可能带 `hivm.tcore_type` 属性 |
| `scope::ScopeOp` | Scope 方言：把一组 op 隔离进一个作用域，是后续 CV 拆分（SeparateCVScope）的最小单元 |
| `CustomOpAnalysis` | 分析一个 custom op：`coreType`（核型）、`gmBuffers`（全局内存 buffer）、`localBuffers`（本地 alloc buffer）、`relevantOps`（需打包的 op）、`isLoad/isUnknownPipe` |
| `kCoreType` | `"ssbuffer.core_type"`：核型属性。**非 hint**——`OpClassifier` 把它当权威，绝不再重新着色 |
| `UNDETERMINED` | 核型"未定"：故意不打 `kCoreType`，让 `OpClassifier` 按数据流自行归类 |
| `packScopeOp` / `unpackScopeOp` | Common 工具：把 op 数组打包进 ScopeOp / 反之拆出（是后续 `UnpackScopePass` 的反向操作） |
| `escapedValues` | 打包后仍被外部使用的值：必须通过 scope 的 terminator（`scope.return`）逃逸，成为 scope 的 result |
| `ToTensorOp` | `bufferization.to_tensor`：memref→tensor 的转换，是 custom op 与 buffer 之间的桥 |

---

## 三、逐行讲解

### 3.1 PackMemrefPass 类（L64-L86）

- `getDependentDialects`：声明 `scope::ScopeDialect`、`hivm::HIVMDialect`、`bufferization::BufferizationDialect`、`memref::MemRefDialect`；
- `getArgument()` = `"ssbuf-standardize-op-pack-memref"`；
- `runOnOperation` 只做一件事：`packAllCustomOpUnits(moduleOp)`。

### 3.2 `runOnOperation` + `packAllCustomOpUnits`（L88-L104）

```cpp
void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  packAllCustomOpUnits(moduleOp);
}

static void packAllCustomOpUnits(ModuleOp moduleOp) {
  SmallVector<hivm::CustomOp> customOps;
  moduleOp.walk([&](hivm::CustomOp customOp) { customOps.push_back(customOp); });
  for (hivm::CustomOp customOp : customOps) {
    packCustomOpUnit(customOp);
  }
}
```

- 先一次性 walk 收集所有 custom op（避免边打包边遍历的迭代器失效），再逐个 `packCustomOpUnit`；
- 每个 custom op 独立打包，互不影响。

### 3.3 `packCustomOpUnit`（L106-L134）

```cpp
static void packCustomOpUnit(hivm::CustomOp customOp) {
  auto anaRes = CustomOpAnalysis::get(customOp);   // 1. 分析
  if (llvm::failed(anaRes)) {                       // 2. 分析失败 → 跳过
    LDBG("warning: failed to analyze " << customOp);
    return;
  }
  auto &ana = anaRes.value();
  auto scopeOp = packScopeOp(ana.relevantOps);     // 3. 打包
  if (!scopeOp) {                                   // 4. 打包失败 → 保持未打包
    // 未打包也安全：buffer 仍是 raw to_tensor 链，
    // OpClassifier 的 matchToTensorPattern 依旧能识别
    LDBG("failed to pack scope: " << customOp << "; fallback to default");
    return;
  }
  // 5. 打 kCoreType（UNDETERMINED 时不打）
  if (ana.coreType != UNDETERMINED) {
    scopeOp->setAttr(kCoreType,
        StringAttr::get(scopeOp->getContext(), coreTypeToString(ana.coreType)));
  }
}
```

**要点**：
- 分析失败或打包失败都**静默降级**（不 fallback、不报错）：保持 custom op 未打包，后续 OpClassifier 仍能通过 raw `to_tensor` 链识别它；
- 打 `kCoreType` 是**权威标记而非 hint**：`OpClassifierPass` 一旦看到已存在的 `ssbuffer.core_type` 就绝不会重新着色。因此 `UNDETERMINED`（= 无法确定核型）时**必须留空**，让 OpClassifier 按数据流分类，否则会把本可优化的 op 错误地钉死在错误核型上。

### 3.4 依赖分析：`CustomOpAnalysis::get`（CustomOpUtils.cpp L171-L183）

```cpp
FailureOr<CustomOpAnalysis> CustomOpAnalysis::get(hivm::CustomOp customOp) {
  CustomOpAnalysis result(customOp);
  result.determineCoreType();              // 从 hivm.tcore_type 推导 coreType
  if (result.collectBuffers().failed())    // 分类 tensor/memref operand
    return failure();
  result.collectRelevantOps();             // 收集打包所需的 op 集
  result.determineLoadOrUnknown();         // 从 pipe 判断 load/store/unknown
  return result;
}
```

- `determineCoreType`（L40-L61）：读 `hivm.tcore_type`——`CUBE`→`CUBE_ONLY`、`VECTOR`→`VECTOR_ONLY`、`CUBE_AND_VECTOR`→`CUBE_AND_VECTOR`；`CUBE_OR_VECTOR`/无属性 → 保持 `UNDETERMINED`（不钉核，让 OpClassifier 决定，避免把 op 拆成 CUBE/VECTOR 两份）。
- `collectBuffers`（L67-L95）：遍历 custom op 的所有 tensor/memref operand，用 `traceMemDef` 溯源到 source memref：若是 `func.func` 的 block argument → 记入 `gmBuffers`（全局内存）；若是 `memref.alloc` → 记入 `localBuffers`（本地 buffer）；溯源到非函数 block argument → `failure`（无法归类，放弃分析）。
- `collectRelevantOps`（L100-L156）：对每个 local buffer 识别两种打包模式，收集要移进 scope 的 op：
  - **只读 buffer**（`alloc → to_tensor → custom`）：`alloc`、`to_tensor`、`custom`（custom 若单结果，则 `to_tensor` 结果直接进 custom）；
  - **填充 buffer**（`alloc → custom → to_tensor`，custom 写 buffer）：`alloc`、`custom`、`to_tensor`；
  - 末尾若输出值单 use 且是 `hivm::ConvertLayoutOp`（layout 转换），也一并收进 `relevantOps`。
- `determineLoadOrUnknown`（L160-L169）：用 `PipeAnalysis(pipe)` 看 pipe 的 `from`/`to` 地址空间——`from` 含 GM → `isLoad`；`to` 含 GM → `isLoad=false`（store）；否则 `isUnknownPipe`。

### 3.5 依赖工具：`packScopeOp`（ScopeOpUtils.cpp L45-L94）

```cpp
scope::ScopeOp packScopeOp(llvm::ArrayRef<Operation *> ops) {
  if (ops.empty()) return nullptr;
  // 1. 找逃逸值：结果被 ops 集合外的用户消费
  llvm::SetVector<Value> escapedValues;
  for (op : ops) for (result : op->getResults())
    for (user : result.getUsers())
      if (!allOps.contains(user)) { escapedValues.insert(result); break; }
  // 2. 建 scope：结果类型 = 逃逸值类型
  auto scopeOp = builder.create<scope::ScopeOp>(loc, escapedValues.getTypes());
  // 3. 先把外部用户重定向到 scope 结果（否则 op 移入 scope 后值不可见）
  for ([origVal, scopeRes] : zip(escapedValues, scopeOp->getResults()))
    origVal.replaceUsesWithIf(scopeRes, [&](OpOperand &o){ return !allOps.contains(o.getOwner()); });
  // 4. 建 body block + scope.return，把 ops 依次 move 进去
  builder.setInsertionPointToEnd(block);
  auto returnOp = builder.create<scope::ReturnOp>(loc, escapedValues);
  for (op : ops) op->moveBefore(returnOp);
  return scopeOp;
}
```

**要点**：
- 先收集逃逸值、再重定向外部 uses、最后把 op 移进 body——顺序很重要，保证移入后原来的 value 仍能从外围 block 通过 scope 的 result 使用；
- `returnOp` 作为 terminator，op 逐个 `moveBefore(returnOp)` 保持相对顺序；
- `unpackScopeOp`（L96-L128）是逆操作（后续 `UnpackScopePass` 会用），先移出 body op 再替换 uses、最后 erase scope。

---

## 四、算法流程总结

```
PackMemrefPass
├─ packAllCustomOpUnits：walk 收集所有 hivm.hir.custom
└─ 对每个 custom op → packCustomOpUnit：
     1. CustomOpAnalysis::get：determineCoreType → collectBuffers → collectRelevantOps → determineLoadOrUnknown
        - 失败 → 跳过（保持未打包）
     2. packScopeOp(relevantOps)：逃逸值 → scope + scope.return → 移入 op
        - 失败 → 保持未打包（raw to_tensor 链仍可识别）
     3. coreType != UNDETERMINED → 打 kCoreType（否则留空给 OpClassifier）
```

复杂度：packAllCustomOpUnits 一次 walk O(N) + 每 custom op 的 operand 溯源 O(operand 链长)。

---

## 五、面试要点

1. **PackMemrefPass 为什么属于 StandardizeOp，而不是 PlanComputeBlock？**
   custom op 打包是"结构化"而非"分类"——它把 op 与 buffer 的关系固化成一个 scope 单元，属于 IR 形态标准化。先在此标准化，`PlanComputeBlock` 的 `OpClassifier` 才能面对干净的自包含单元做核型分类。

2. **为什么打包失败要静默降级（不 fallback）？**
   打包是"锦上添花"的优化，不是正确性前提。未打包时 custom op 仍持 raw `to_tensor` 链，`OpClassifier::matchToTensorPattern` 依然能识别并分类；硬要 fallback 反而会牺牲整个 CV 流水线。

3. **为什么 `UNDETERMINED` 时故意不打 `kCoreType`？**
   `kCoreType`（`ssbuffer.core_type`）在 `OpClassifier` 眼里是**权威**——一旦存在就不会再重算。若把一个本应"按数据流决定核型"的 op 钉死在 `UNDETERMINED`，等于剥夺了 OpClassifier 的决策权；留空则让 OpClassifier 自由分类，这也是注释里强调"stamping is not a hint"的原因。

4. **custom op 的 buffer 为什么要区分 GM / local？**
   这决定了打包范围：GM buffer（func 参数）是跨核共享的全局内存，**不**随 custom op 打包进 scope（它不属于局部计算）；local buffer（`memref.alloc`）才是 custom op 私有的、需要一起打包进 scope 的缓冲。`CustomOpAnalysis::collectBuffers` 据此只把 local buffer 的构建链 `collectRelevantOps` 收进待打包集。

5. **`packScopeOp` 里"先重定向外部 uses 再移入 op"的顺序为什么关键？**
   若先移入 op，原 value 的定义点就进了 scope body，外围 block 无法再直接引用；所以必须先用 `replaceUsesWithIf` 把外部 uses 改指向 scope 的 result，再移动 op。逃逸值通过 `scope.return` 成为 scope result，内部计算对外表现为一个黑盒单元。

6. **PackMemrefPass 与后续别处的关系？**
   它是 `scope::ScopeOp` 生成的第一个来源之一（后续 SeparateCVScope 会基于 scope 做 CUBE/Vector 拆核）。`unpackScopeOp` 是它的逆操作，供后续 `UnpackScopePass` 把不再需要的 scope 拆开，两个方向共同维护 scope 结构的生命周期。