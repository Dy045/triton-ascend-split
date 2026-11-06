# SplitDataflow 子 Pass 专题（九）：UnpackScope —— scope 拆包还原

> 源码文件：[`UnpackScopePass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/UnpackScopePass.cpp)（118 行，**main-dev 新增**）
>
> 流水线位置：源码位于 SplitDataflow 目录，但**由 `ComputeBlockOptPass` 在收尾阶段 E 的 L91 调用**（`createUnpackScopePass`），与 `StandardizeOp` 的 `PackMemrefPass`（打包 scope）形成一对"打包/拆包"。
>
> Pass 名：`ssbuf-unpack-scopeop`（`getArgument()`）。

---

## 一、这个 Pass 在做什么

`StandardizeOp` 的 `PackMemrefPass`（见 `StandardizeOp_03_PackMemrefPass`）在流水线前段把 `hivm.hir.custom` + 本地 `memref.alloc` + `to_tensor` 链打包进一个 `scope::ScopeOp`，作为"原子单元"参与后续分块/传输/同步。到了 ComputeBlockOpt 收尾阶段（`SplitIfByBlockId` 等已经完成），这个原子单元不再需要，`UnpackScopePass` 把它**拆回**原本的独立 op 形态，交还给标准编译链。

`unpackScopeOp` 会把 scope body 里的 op 移出来、删掉 scope op；本 Pass 在拆包之后还要做一个关键动作——**把 scope 上的属性（尤其是 `ssbuffer.block_id`）回填到被拆出来的每个 op 上**。

> 为什么属性要回填：scope body 内的 op 在打包期间没有被单独打 `block_id`（打标都刻在 scope 层），拆包后这些 op 若不带 `block_id`，后续依赖 `block_id` 的 pass 就会失效。所以拆包必须把 scope 携带的属性"继承"给 body 内的 op。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `scope::ScopeOp` | 打包单元 op，`PackMemrefPass` 用它把 custom+alloc+to_tensor 链包成一个原子单元 |
| `unpackScopeOp` | `Common/ScopeOpUtils.h` 提供的工具：把 scope body 的 op 移出、删掉 scope，返回被移出的 op 列表 |
| 属性复用（AttributeArray 共享） | body op 属性为空、或只有 `block_id` 时，可直接共享 scope 的整份 `AttributeArray`（在 MLIRContext 里全局共享，不逐属性拷贝） |
| `kBlockId` / `ssbuffer.block_id` | 计算块编号属性，拆包后必须继承 |
| 先收集后拆包 | 避免在 `walk` 回调里 mutate IR 导致遍历栈失效 |

---

## 三、逐行讲解

### 3.1 Pass 定义与依赖（L49-L107）

```cpp
class UnpackScopePass
    : public PassWrapper<UnpackScopePass, OperationPass<ModuleOp>> {
  void runOnOperation() override { ... }

  llvm::StringRef getArgument() const final { return "ssbuf-unpack-scopeop"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scope::ScopeDialect>();   // 依赖 scope 方言
  }
};
```

- Pass 是 `OperationPass<ModuleOp>`（作用在整个 Module）；`getDependentDialects` 声明依赖 `scope::ScopeDialect`（要识别 `scope::ScopeOp` 必须加载该方言）。

### 3.2 `runOnOperation`（L56-L98）

```cpp
void runOnOperation() override {
  auto module = getOperation();
  if (hasFallbackAttr(module)) return;   // fallback 兜底

  // Step 1: 先收集全部 ScopeOp，再统一拆包
  SmallVector<scope::ScopeOp> scopes;
  module.walk([&](scope::ScopeOp scopeOp) { scopes.push_back(scopeOp); });

  for (scope::ScopeOp scopeOp : scopes) {
    // Step 2: 拆包（把 body op 移出、删掉 scope）
    SmallVector<Operation *> unpackedOps;
    auto attrs = scopeOp->getAttrs();
    if (unpackScopeOp(scopeOp, &unpackedOps).failed()) {
      setFallbackAttr(module, ERRCODE_FAILED);
      return;
    }

    // Step 3: 把 scope 的属性回填给每个拆出的 op
    for (auto *op : unpackedOps) {
      auto canReuseAttrs = op->getAttrs().size() == 0;         // 空属性 → 整体共享
      if (op->getAttrs().size() == 1) {
        canReuseAttrs = op->hasAttr(kBlockId);                  // 只有 block_id → 也可共享
      }
      if (canReuseAttrs) {
        op->setAttrs(attrs);                                    // 共享整个 AttributeArray
      } else {
        for (const auto attr : attrs)
          op->setAttr(attr.getName(), attr.getValue());         // 逐属性拷贝
      }
    }
  }
}
```

**逐点讲解**：

- **先收集、后拆包（L66-L67）**：`unpackScopeOp` 会把 body op 移出并删除 scope，如果在 `module.walk` 的回调里就地 mutate，会破坏 walk 自身的遍历栈。所以先 `walk` 收集所有 scope 指针，收集完再统一拆。默认 post-order 会先访问内层 scope、再访问外层 scope；拆内层 scope 只会把 op 移出外层 body（不会让外层 scope 失效），因此收集到的每个指针在拆包时都仍然有效（这也是收集阶段不需要额外判空的依据）。

- **属性回填（L77-L96）**：被拆出的 op 在打包期间没有自己的 `block_id`（属性都刻在 scope 层），拆包后必须把 scope 的属性继承下来。继承分两种：
  1. **可复用整份 AttributeArray**（`canReuseAttrs`）：op 属性为空、或只有 `block_id` 一个属性时，它的属性集是 scope 属性的子集，可以直接 `op->setAttrs(attrs)` 共享 scope 的整份 `AttributeArray`（AttributeArray 在 MLIRContext 里是全局共享的，`setAttrs` 只是让 op 指向同一份，不产生逐属性拷贝）——省内存；
  2. **逐属性拷贝**：op 已有多个属性（如还带了 `ssbuffer.core_type`），则逐条 `setAttr` 覆盖/补充，避免整份 setAttrs 把 op 已有的无关属性冲掉。
  - **为什么 scope 一定带 `block_id`**：`PackMemrefPass` 打标时把 `block_id` 刻在 scope 层；若 scope 没有 `block_id`，是上游 bug，由后续消费 `block_id` 的 pass 报错。

---

## 四、算法流程总结

```
UnpackScopePass
├─ fallback 兜底检查
├─ module.walk 收集所有 scope::ScopeOp（先收集，不 mutate）
└─ 逐个 scope：
    ├─ unpackScopeOp 拆包（body op 移出、scope 删除）→ 失败则 fallback
    └─ 属性回填：
        ├─ op 属性为空 / 只有 block_id → 共享 scope 的整个 AttributeArray
        └─ 否则 → 逐属性 setAttr 拷贝
```

**整体策略**：这是 `PackMemrefPass` 的逆操作，属于典型"打包→分块调度→拆包还原"的资源模式——打包让 custom+alloc+to_tensor 在分块/传输阶段作为一个原子单元被统一调度，收尾再拆回独立 op 并把属性正确继承回 body op，保证后续标准编译链（以及依赖 `block_id` 的降级/调试 pass）仍然能看到一致的分块信息。

---

## 五、面试要点

1. **为什么要先 `PackMemref` 打包、最后又 `UnpackScope` 拆包？**
   分块/传输/同步阶段，`custom + alloc + to_tensor` 这条链要作为一个整体参与调度（共享同一个 `block_id`、一起被传输/同步），打包成 `scope::ScopeOp` 让它们"原子化"，避免拆散后被错误的块合并/传输打断；到收尾阶段这些约束已经不需要了，拆回独立 op 交还标准编译链。

2. **为什么拆包后要回填属性？**
   打包期间 `block_id` 等属性刻在 scope 层，body 内的 op 没有单独打标；拆包后若不把 scope 的属性继承回 body op，后续依赖 `block_id` 的 pass（调度、降级、调试）就看不到正确的分块信息，甚至失效。

3. **"先收集后拆包"避免了什么问题？**
   `unpackScopeOp` 会删除 scope 并移动 body op，在 `walk` 回调里就地 mutate 会破坏 walk 的遍历栈（迭代器失效/漏遍历）。先 `walk` 收集全部 scope 指针，再统一拆，配合默认 post-order（内层 scope 先拆、只往外移 op）保证指针全程有效。

4. **为什么"空属性/只有 block_id"的 op 可以直接共享整份 AttributeArray？**
   这两种情况下 op 的属性集是 scope 属性集的子集，把 op 指向 scope 的同一份 `AttributeArray` 是安全且更省内存的（AttributeArray 在 MLIRContext 全局共享）。反之若 op 已有多个属性，整份 setAttrs 会冲掉它自己的无关属性，只能逐属性 `setAttr` 覆盖。

5. **这个 Pass 为什么源码在 SplitDataflow 目录，却在 ComputeBlockOpt 里被调用？**
   `UnpackScopePass` 是 `PackMemrefPass` 的逆操作，而 `PackMemrefPass` 在 `StandardizeOp`（更靠近 SplitDataflow 的职责域）。拆包逻辑与"scope 打包工具、CoreType 分类"等 SplitDataflow/Common 的能力强相关，因此实现放在 SplitDataflow 目录复用这些工具；但它实际被编排的位置在 `ComputeBlockOptPass` 收尾阶段 E（`SplitIfByBlockId` 之后），因为只有到那时 scope 原子单元才完成使命。这反映了"实现按能力归属、调用按编排顺序"的组织方式。