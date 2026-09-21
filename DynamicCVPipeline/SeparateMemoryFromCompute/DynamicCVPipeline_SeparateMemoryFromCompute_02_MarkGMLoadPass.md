# DynamicCVPipeline 主 Pass 专题：访存与计算分离 · 子专题 02 —— MarkGMLoadPass（GM 加载多缓冲标记）

> 覆盖 `SeparateMemoryFromCompute` 的 **子 Pass 1：MarkGMLoadPass**（独立 Pass）
>
> 源码文件：[`MarkGMLoadPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SeparateMemoryFromCompute/MarkGMLoadPass.cpp)、[`MarkGMLoadPass.h`](../../../third_party/ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/MarkGMLoadPass.h)
>
> 流水线位置：`SeparateMemoryFromCompute`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L84）内第一个子 Pass（主编排 L47）。上游条件流水线已成型，本 Pass 把 GM 加载路径标记为可多缓冲，供后端解耦访存与计算。

---

## 一、这个子 Pass 在做什么

多缓冲流水线要真正转起来，**数据搬运（GM→UB 的 `memref.copy`）也必须参与重叠**——否则计算侧多缓冲了、加载还是单缓冲串行，访存延迟依旧暴露。

`MarkGMLoadPass` 的任务：**找出所有"从全局内存加载到 UB 缓冲"的拷贝，并在目标 `memref.alloc` 上打多缓冲标记（`annotation.mark + hivm.multi_buffer = N`）**，让后端知道这块缓冲该开几层。

识别与标记分三步：

1. **Rule 1（源校验）**：`memref.copy` 的源必须能穿透 view-like / `tensor.extract_slice` / `scf.for` iter_arg / `scf.while` 前后块，最终追溯到 **entry 函数的 BlockArgument**——即 GM 指针。不是 GM 指针的拷贝（如核间搬运）跳过；
2. **Rule 2（目标校验）**：目标链穿透 view-like / `tensor.extract_slice` 后必须是 `memref::AllocOp`（本地 UB 缓冲），否则没有可挂标记的载体；
3. **Rule 3（深度确定）+ 打标记**：缓冲深度 N 优先取**用户编译 hint**（`gm_load` 属性：0/1 强制关闭、≥2 强制开启指定深度），否则按 scope 类型取默认（Vector=2、CUBE=1）；随后在 destAlloc 后插入（或更新已存在的）`annotation.mark`，写上 `multi_buffer=N`，hint 来源额外打 `gm_load_hint` 标记（供下游裁剪时识别保护区）。

**一句话**：这是"访存与计算分离"的**使能标记 Pass**——它不搬 op，只把 GM 加载缓冲"升级"为多缓冲并留下 hint 痕迹。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `MarkCandidate` | 候选结构 `{copyOp, destAlloc, scopeOp, bufferCount, fromHint}`：copyOp 与目标 alloc、所属 scope；bufferCount/fromHint 在 Phase 2 填充 |
| `kDefaultVBufferCount=2` / `kDefaultCBufferCount=1` | Vector/CUBE scope 的默认缓冲深度（L55-L56） |
| `resolveFuncBlockArg` | 判断 func 参数是否为 entry 参数：entry（`symbolKnownUseEmpty`）→ GM 源；非 entry → 沿 caller 实参继续追溯（L72-L93） |
| Rule 1 `traceSourceToFuncArg` | 源链回溯，见 3.2（L98-L156） |
| Rule 2 `traceDestToAlloc` | 目标链回溯到 alloc，见 3.3（L161-L174） |
| Rule 3 `resolveBufferCount` | 按 scope 类型返回默认深度；scope 为空或 `tcore_type` 异常返回 -1（L179-L196） |
| `resolveHintBufferCount` | 用户 hint 解析：`gm_load` IntegerAttr；hint 冲突返回 -1；hint 属性被消费后移除（L206-L235） |
| `kGMLoadMultiBufferHintAttr="gm_load"` | 用户 hint 属性名（Common/Utils.h L81） |
| `kGMLoadHintAttr="gm_load_hint"` | 标记来源标记（UnitAttr），表示此多缓冲深度来自用户 hint（Common/Utils.h L82） |
| `markGMLoadCandidate` | 实际打标记：bufferCount≤1 跳过；已有 markOp 则更新属性，否则新建 markOp（L257-L288） |
| `hivm::MultiBufferAttr::name` | 后端识别的多缓冲深度属性（`annotation.mark` 上携带） |
| Phase1/Phase2/Phase3 | 只读收集 → 解析深度 → 打标记（L308-L345） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（MarkGMLoadPass.cpp L299-L348）

```cpp
void MarkGMLoadPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  // Phase 1：只读收集候选（不修改 IR）
  SmallVector<MarkCandidate, 8> candidates;
  module.walk([&](memref::CopyOp copyOp) {
    if (auto candidate = collectCandidate(copyOp)) {
      candidates.push_back(*candidate);
    }
  });
  if (candidates.empty()) { LOG_DEBUG("no GM load candidate found"); return; }

  // Phase 2 & 3：逐候选解析深度并打标记
  for (auto &c : candidates) {
    int hintVal = resolveHintBufferCount(c.destAlloc);
    if (hintVal >= 0) {                 // 用户指定 hint 优先
      if (hintVal <= 1) continue;       // 0/1 = 强制关闭，跳过
      c.bufferCount = hintVal; c.fromHint = true;
    } else {                            // 无 hint：自动解析
      c.bufferCount = resolveBufferCount(c.scopeOp);
      if (c.bufferCount < 0) {
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return;
      }
    }
    markGMLoadCandidate(c);             // 写 annotation.mark + multi_buffer
  }
}
```

**要点**：
- **先收集后改写**：Phase 1 walk 只读收集（避免 walk 中改 IR），Phase 2/3 再统一改写——符合"先分析后变换"的稳健模式；
- **hint 优先级高于默认值**：用户显式指定深度时直接采用（≥2），0/1 直接放弃该候选；无 hint 才走 scope 默认值；
- **自动解析失败置 fallback**：scope 类型异常（无 `tcore_type`）属结构性错误，置 `ERRCODE_FAILED` 让上游回退。

### 3.2 Rule 1：源链回溯 traceSourceToFuncArg（L98-L156）

```cpp
static bool traceSourceToFuncArg(Value v) {
  while (true) {
    // 1. 穿透 view-like op 与 tensor.extract_slice
    while (auto *defOp = v.getDefiningOp()) {
      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) { v = viewLike.getViewSource(); continue; }
      if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) { v = extractSlice.getSource(); continue; }
      break;
    }
    // 2. 终点必须是 BlockArgument
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg) return false;
    Operation *parentOp = blockArg.getOwner()->getParentOp();

    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {      // 函数参数
      Value nextV;
      if (resolveFuncBlockArg(blockArg, funcOp, nextV)) return true; // entry 参数 → GM 源
      if (nextV) { v = nextV; continue; }                            // 非 entry → 沿 caller 实参
      return false;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg.getArgNumber() == 0) return false;          // 归纳变量不能是 GM 源
      v = forOp.getInitArgs()[blockArg.getArgNumber() - 1];    // iter_arg → 其 init 值
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (blockArg.getOwner() == whileOp.getBeforeBody()) {    // before 块 arg → inits
        v = whileOp.getInits()[blockArg.getArgNumber()]; continue;
      }
      if (blockArg.getOwner() == whileOp.getAfterBody()) {     // after 块 arg → condition 实参
        v = condOp.getArgs()[blockArg.getArgNumber()]; continue;
      }
      return false;
    }
    return false;   // 其他 BlockArgument 类型不支持
  }
}
```

**要点**：
- **view-like 穿透**：`subview`/`view`/`tensor.extract_slice` 只是"视角"，源仍是底层 buffer，必须剥掉；
- **for 的 iter_arg**：循环内 copy 的源常是前一迭代的迭代变量（流水线轮换），取其 **init 值**继续追溯；arg 0 是归纳变量（int），不可能指向 GM，直接拒绝；
- **while 双向回溯**：before 块 arg 取 `inits`，after 块 arg 取 `scf.condition` 的实参——after 块的值来自条件分支的参数；
- **非 entry 函数**：`resolveFuncBlockArg` 用 `symbolKnownUseEmpty` 判断是否入口函数，非入口则取第一个 caller 的对应实参继续追（内联前的函数间追溯）。

#### resolveFuncBlockArg（L72-L93）

```cpp
static bool resolveFuncBlockArg(BlockArgument blockArg, func::FuncOp funcOp, Value &nextValue) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (SymbolTable::symbolKnownUseEmpty(funcOp.getNameAttr(), moduleOp))
    return true;                        // 入口函数参数 → 外部传入的 GM 指针
  auto symbolUses = SymbolTable::getSymbolUses(funcOp.getNameAttr(), moduleOp);
  if (symbolUses && !symbolUses->empty()) {
    auto callOp = cast<func::CallOp>((*symbolUses->begin()).getUser());
    nextValue = callOp.getOperands()[blockArg.getArgNumber()];  // 沿调用实参继续
    return false;
  }
  return false;                         // 无 caller：追溯失败
}
```

### 3.3 Rule 2：目标链回溯 traceDestToAlloc（L161-L174）

```cpp
static memref::AllocOp traceDestToAlloc(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(defOp)) { v = viewLikeOp.getViewSource(); continue; }
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) { v = extractSliceOp.getSource(); continue; }
    break;
  }
  return dyn_cast_or_null<memref::AllocOp>(v.getDefiningOp());  // 必须终止于 alloc
}
```

**要点**：目标必须是 `memref.alloc`（本地 UB 缓冲）。若目标链终止于其他 op（如核间 buffer、`tensor.empty` 的 bufferization 结果），说明该拷贝不是"GM→本地"形态，跳过。与 Rule 1 的穿透规则一致。

### 3.4 collectCandidate（L239-L252）

```cpp
static std::optional<MarkCandidate> collectCandidate(memref::CopyOp copyOp) {
  // Rule 1：源必须追溯到函数参数（GM 指针）
  if (!traceSourceToFuncArg(copyOp.getSource())) return std::nullopt;
  // Rule 2：目标必须由 memref::AllocOp 创建
  auto allocOp = traceDestToAlloc(copyOp.getTarget());
  if (!allocOp) { LOG_DEBUG("dest not created by memref::AllocOp, skip"); return std::nullopt; }
  auto scopeOp = copyOp->getParentOfType<scope::ScopeOp>();
  return MarkCandidate{copyOp, allocOp, scopeOp};
}
```

**要点**：两规则都过才构成候选；`scopeOp` 用 `getParentOfType` 就近取（Phase 3 决定缓冲深度用）。

### 3.5 Rule 3：resolveBufferCount（L179-L196）

```cpp
static int resolveBufferCount(scope::ScopeOp scopeOp) {
  if (!scopeOp) return -1;
  bool isCube = false, isVector = false;
  if (failed(getScopeType(scopeOp, isCube, isVector))) return -1;
  if (isVector)      return kDefaultVBufferCount;   // 2：Vector 天然需要双缓冲隐藏访存延迟
  else if (isCube)   return kDefaultCBufferCount;   // 1：CUBE 通常单缓冲即可
  return -1;
}
```

**要点**：默认深度与 `AddControlFlowCondition` 的缓冲分配策略呼应——Vector 侧多缓冲（2）、CUBE 侧单缓冲（1）。返回 -1 只在 scope 缺失/类型异常时发生。

### 3.6 resolveHintBufferCount（L206-L235）

```cpp
static int resolveHintBufferCount(memref::AllocOp destAlloc) {
  Value allocResult = destAlloc.getResult();
  std::optional<int> foundHint;
  auto hasConflictingHint = [&](annotation::MarkOp markOp) -> bool {
    auto attr = markOp->getAttrOfType<IntegerAttr>(CVPipeline::kGMLoadMultiBufferHintAttr); // "gm_load"
    if (!attr) return false;
    int val = attr.getInt();
    markOp->removeAttr(CVPipeline::kGMLoadMultiBufferHintAttr);  // 消费 hint
    if (foundHint && *foundHint != val) return true;             // 冲突
    foundHint = val; return false;
  };
  for (auto *user : allocResult.getUsers()) {
    // 后 bufferization：hint 在 to_tensor 结果上
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
      for (auto *tensorUser : toTensor.getResult().getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(tensorUser)) {
          if (hasConflictingHint(markOp)) return -1;
        }
      }
    }
  }
  return foundHint.value_or(-1);   // 无 hint 返回 -1
}
```

**要点**：
- hint 属性 `gm_load` 可能打在 alloc 上（bufferization 后）或 `to_tensor` 产物上（bufferization 前），两种形态都要找；
- **读取即消费**：hint 解析后 `removeAttr`，防止被后端重复读取/误读；
- **冲突即弃用**：同一 alloc 出现多个不同 hint 值 → 返回 -1，走默认值（不信任矛盾指令）。

### 3.7 markGMLoadCandidate（L257-L288）

```cpp
static bool markGMLoadCandidate(MarkCandidate &c) {
  if (c.bufferCount <= 1) return false;                 // 单缓冲无需标记
  // 找已存在的 markOp（幂等更新）
  for (auto *user : c.destAlloc->getUsers())
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) { existingMarkOp = markOp; break; }
  OpBuilder builder(c.destAlloc);
  if (existingMarkOp) {
    existingMarkOp->setAttr(hivm::MultiBufferAttr::name, builder.getI32IntegerAttr(c.bufferCount));
    if (c.fromHint) existingMarkOp->setAttr(CVPipeline::kGMLoadHintAttr, builder.getUnitAttr());
  } else {
    builder.setInsertionPointAfter(c.destAlloc);
    auto markOp = builder.create<annotation::MarkOp>(c.destAlloc->getLoc(), c.destAlloc.getResult());
    markOp->setAttr(hivm::MultiBufferAttr::name, builder.getI32IntegerAttr(c.bufferCount));
    if (c.fromHint) markOp->setAttr(CVPipeline::kGMLoadHintAttr, builder.getUnitAttr());
  }
  return true;
}
```

**要点**：
- **bufferCount≤1 不标记**：深度 1 等同单缓冲，打标记无意义（也为下游 `collectBuffers` 过滤掉平凡情况）；
- **已有 markOp 则更新而非新建**：`annotation.mark` 可能由前序 Pass（如 `AllocMultiCache`）打过，保持幂等；
- **`gm_load_hint` 是保护区标识**：UBOverflowChecker 的 `collectPruneCandidates` 明确排除 `fromHint` 缓冲，用户强制指定的深度永不被自动裁剪。

---

## 四、流程总结

### 4.1 处理流水

```
MarkGMLoadPass（runOnOperation L299）
  ├─► fallback 短路（L302）
  ├─► Phase1 只读收集（L309-L314）
  │      module.walk memref.copy
  │        ├─► collectCandidate：Rule1 源→entry func 参数（GM 指针）
  │        │     穿透 view-like / extract_slice / for-iter_arg(init) / while(before→inits, after→condition args)
  │        └─► Rule2 目标→穿透后必须 memref.alloc
  ├─► Phase2 解析缓冲深度（L323-L343）
  │      ├─► resolveHintBufferCount：gm_load hint（0/1 关、≥2 强制开、冲突-1）
  │      └─► 无 hint → resolveBufferCount：Vector=2 / CUBE=1，失败 → ERRCODE_FAILED
  └─► Phase3 打标记（L344）
         └─► markGMLoadCandidate：destAlloc 后插/更新 annotation.mark + multi_buffer=N
               + fromHint → gm_load_hint（保护区）
```

### 4.2 与上下游的协作

- **输入**：`memref.copy`（GM→UB）、`memref.alloc`、`scope::ScopeOp`（tcore_type）、用户 `gm_load` hint 属性；
- **产出**：GM 加载缓冲上的 `annotation.mark + multi_buffer`（+ `gm_load_hint`）；
- **下游**：`UBOverflowCheckerPass` 统计并裁剪超限标记；后端据此决定缓冲深度，实现访存与计算重叠。

---

## 五、面试要点

1. **MarkGMLoadPass 的目标？** 找出所有"GM→本地 UB"的 `memref.copy`，在目标 alloc 上打 `multi_buffer=N` 标记，使 GM 加载路径与计算重叠（多缓冲化），实现"访存与计算分离"。

2. **为什么源必须追溯到函数参数？** 只有 entry 函数的指针参数才是 GM 指针。核间 buffer、子块内生成的数据都不来自 GM，打多缓冲标记会误导后端分配。

3. **为什么循环 iter_arg 取 init 值继续追溯？** 流水线轮换后循环内 copy 的源常是前一轮迭代的迭代变量，它不是 GM 指针本身；取其 **init 值**（循环外的原始值）才能找到真正的 GM 指针。for 的 induction variable（arg 0）是整型计数，直接排除。

4. **view-like 穿透是干什么？** `subview`/`view`/`extract_slice` 不产生新数据，只是数据视角；源/目标链必须剥掉这些"马甲"才能看到底层 buffer/alloc 本体。

5. **用户 hint 的语义？** `gm_load` 属性：0/1 = 强制关闭多缓冲（跳过）；≥2 = 强制开启且指定深度；多 hint 冲突则弃用走默认。hint 强制开的标记打 `gm_load_hint`，下游裁剪时跳过——用户显式意图优先于自动优化。

6. **默认深度为什么 Vector=2、CUBE=1？** Vector 访存延迟显著，双缓冲能隐藏；CUBE 计算密集、访存压力相对小，单缓冲足够。与 `AddControlFlowCondition`/`AllocMultiCache` 的分配策略一致。

7. **为什么 bufferCount≤1 不标记？** 深度 1 就是单缓冲，标记无收益；跳过还能让下游 `collectBuffers`/`pruneMultiBufferMarks` 只看到真正需要优化的对象。

8. **为什么已有 markOp 时更新而非新建？** `annotation.mark` 可能已由前序 Pass（`AllocMultiCache`）创建；直接更新属性保证幂等，不产生重复 mark。

9. **失败处理为何置 ERRCODE_FAILED？** `resolveBufferCount` 失败意味着 scope 类型异常（结构性问题），不是"少优化"而是"IR 不符合预期"，必须触发上游 fallback；而找不到候选/单个候选不成立只是"没得优化"，正常返回。
