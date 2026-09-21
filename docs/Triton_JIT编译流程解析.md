# Triton（Ascend）算子 JIT 编译流程解析

> 本文结合 triton-ascend 仓库源码，说明"写一个算子"背后从 Python 源码到最终可执行二进制的完整编译流程，
> 并对比上游 Triton（NVIDIA 后端）与 triton-ascend（昇腾 NPU 后端）两种架构的差异。

---

## 1. 背景：写一个算子意味着什么

写一个算子，本质上是用 Python 写一个被 `@triton.jit` 装饰的 kernel 函数，描述"**一个 program 如何并行处理一小块数据**"（SPMD 编程模型），然后由编译器把它编译成能在昇腾 NPU 上运行的机器码。它由三层组成：

| 层 | 内容 | 对应代码 |
|---|---|---|
| 算法层 | 描述单个 program 的行为：加载 → 计算 → 存储 | kernel 函数体（`tl.load` / `tl.store` 等） |
| 调度层 | 决定并行规模：grid、`BLOCK_SIZE` 等编译期常量 | 包装函数 + `add_kernel[grid](...)` |
| 编译层 | 把高级语言映射到硬件：IR 生成、优化、代码生成 | Triton 编译器（前端 + 后端） |

以教程 [01-vector-add.py](file:///c:/main/triton-ascend/python/tutorials/01-vector-add.py#L29-L54) 为例：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                 # 我是第几个 program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                 # 越界保护
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 2. 前置：`@triton.jit` 装饰器

`@triton.jit` 装饰器定义在 [python/triton/runtime/jit.py](file:///c:/main/triton-ascend/python/triton/runtime/jit.py)，返回一个 `JITFunction` 对象（继承自 `JITCallable`）。

- **装饰那一刻只创建对象，不编译**。
- `JITCallable.__init__`（[jit.py#L455-L528](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L455-L528)）完成源码获取与预处理。
- 真正的语义分析、类型推断、符号表构建发生在**首次调用触发的编译期**。

---

## 3. 阶段一：源码获取与 AST 解析（runtime/jit.py）

AST 解析入口**不在** `compiler/ast.py`（该文件不存在），而在 `runtime/jit.py`：

1. **取源码**：`inspect.getsourcelines(fn)`（[jit.py#L461](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L461)）
2. **预处理**：`textwrap.dedent` 去掉缩进，再用正则去掉 `@triton.jit` 装饰器行（[jit.py#L468-L469](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L468-L469)）
3. **解析**：`ast.parse`，在 `JITCallable.parse()`（[jit.py#L523-L528](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L523-L528)）

例如 `add_kernel` 会被解析成如下 AST 结构（示意）：

```python
FunctionDef(name='add_kernel', args=[x_ptr, y_ptr, output_ptr, n_elements,
                                      BLOCK_SIZE (annotation=TLConstExpr)],
            decorator_list=[Name('triton.jit')],
            body=[
                Assign(pid = Call(tl.program_id, axis=0)),
                Assign(block_start = BinOp(pid * BLOCK_SIZE)),
                Assign(offsets = BinOp(block_start + Call(tl.arange, 0, BLOCK_SIZE))),
                Assign(mask = Compare(offsets < n_elements)),
                Assign(x = Call(tl.load, x_ptr + offsets, mask=mask)),
                Assign(y = Call(tl.load, y_ptr + offsets, mask=mask)),
                Assign(output = BinOp(x + y)),
                Expr(Call(tl.store, output_ptr + offsets, output, mask=mask)),
            ])
```

### DependenciesFinder（依赖分析与哈希）

`DependenciesFinder` 继承 `ast.NodeVisitor`，定义在 [jit.py#L34](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L34)，职责：

- **依赖分析**：追踪函数及其间接调用的全局变量、非局部变量、其他 JIT 函数；
- **哈希生成**：基于源码与依赖生成稳定 cache_key，源码或依赖变化时自动使缓存失效；
- **一致性检查**：记录函数用到的全局变量值，运行时校验未变（否则报错要求重编译）。

它在 `JITCallable.cache_key` 属性中被调用（[jit.py#L494-L515](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L494-L515)）。

> 注意：`DependenciesFinder` 只做依赖收集与哈希，**不做类型分析**。类型分析在下一阶段由 `CodeGenerator` 完成。

---

## 4. 阶段二：首次调用的编译触发链

第一次调用 `add_kernel[grid](...)` 时的完整链路：

```
add_kernel[grid](...)
  └─ JITFunction.__getitem__   (jit.py#L364：记住 grid，返回启动代理)
       └─ run()                (jit.py#L695：binder 生成签名/specialization，
                                 _pack_args 提取 constexpr/attrs)
            └─ _do_compile()   (jit.py#L826：构造 ASTSource)
                 └─ triton.compiler.compile  (compiler/compiler.py#L228：查磁盘缓存，未命中则编译)
                      └─ ASTSource.make_ir() (compiler/compiler.py#L80)
                           └─ ast_to_ttir()  (compiler/code_generator.py#L1659：AST → TTIR)
                                └─ backend.add_stages() 各阶段（ttir → ttadapter → npubin）
                                     └─ kernel.run() 启动
```

**缓存机制**：
- 内存缓存：按 `(device, key)` 存于 `kernel_cache`（[jit.py#L707-L713](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L707-L713)）；
- 磁盘缓存：`FileCacheManager`（[compiler.py#L250](file:///c:/main/triton-ascend/python/triton/compiler/compiler.py#L250)），key 由源码哈希、签名、constexpr、后端选项、环境变量共同决定。

---

## 5. 阶段三：AST → TTIR（compiler/code_generator.py）

文件存在（✅）。核心：

- 入口函数 [ast_to_ttir](file:///c:/main/triton-ascend/python/triton/compiler/code_generator.py#L1659)；
- `CodeGenerator` 类继承 `ast.NodeVisitor`（[code_generator.py#L303](file:///c:/main/triton-ascend/python/triton/compiler/code_generator.py#L303)），`visit_FunctionDef` 在 [L603](file:///c:/main/triton-ascend/python/triton/compiler/code_generator.py#L603)；
- 在此完成**语义分析 / 类型推断 / 符号表（gscope）构建**、`tl.*` 语言内置的解析与代码生成。

**IR 与类型从哪来**（注意：没有 `compiler/ir.py`）：
- Python 侧操作 MLIR 的句柄绑定来自 C++ 扩展：`triton._C.libtriton.ir`（[compiler.py#L4](file:///c:/main/triton-ascend/python/triton/compiler/compiler.py#L4)、[code_generator.py#L22](file:///c:/main/triton-ascend/python/triton/compiler/code_generator.py#L22)）；
- 本仓库还新增 `triton._C.libtriton.ascend.ir`（ascend_ir）与 `buffer_ir` 方言；
- 语言层类型（`tensor`、`base_value`、`base_type`、`constexpr` 等）定义在 [python/triton/language/core.py](file:///c:/main/triton-ascend/python/triton/language/core.py)；
- MLIR Dialect 定义在 C++ 侧 [include/triton/Dialect](file:///c:/main/triton-ascend/include/triton/Dialect) 与 [lib/Dialect](file:///c:/main/triton-ascend/lib/Dialect)。

---

## 6. 两种架构对比：编译链

### 6.1 上游 Triton（NVIDIA / AMD 后端）

```
Python 源码
  ├─ inspect/ast 解析（runtime/jit.py：JITFunction / JITCallable / DependenciesFinder）
  ├─ code_generator.py：ast_to_ttir ──▶ TTIR（硬件无关、block 级张量）
  ├─ TTGIR（TritonGPU dialect：引入 warp / shared memory / layout / tensor core）★
  ├─ LLVM IR
  ├─ PTX (NVIDIA) / AMDGCN (AMD)
  └─ SASS / cubin / hsaco
```

### 6.2 triton-ascend（昇腾 NPU 后端，本仓库）

由 [AscendBackend.add_stages](file:///c:/main/triton-ascend/third_party/ascend/backend/compiler.py#L1225-L1246) 定义：

```
Python 源码
  ├─ inspect/ast 解析（runtime/jit.py：与上游同一份代码）✅ 一致
  ├─ code_generator.py：ast_to_ttir ──▶ TTIR（含 ascend patch：min_dot_size、extension dispatch）✅ 一致
  ├─ TTAdapter（Linalg IR，ttir_to_linalg）★ 没有 TTGIR！
  ├─ （字节码模式额外：mlirbc ──bishengir-opt──▶ bcmlir）
  └─ NPUBin（昇腾可执行二进制）
```

**为什么 Ascend 没有 TTGIR**：TTGIR 的核心是 warp / thread / shared memory / MMA layout 等 GPU 概念。昇腾是 AIC（Cube/向量）+ AIV（标量）异构核结构，没有 warp 与共享内存模型，因此在 TTIR 之后直接用 `triton_to_linalg` 系列 pass 降到 Linalg（[ttir_to_linalg](file:///c:/main/triton-ascend/third_party/ascend/backend/compiler.py#L155-L258)），再交由昇腾编译器生成 NPU 二进制。

### 6.3 逐条对照表

| 你的原描述 | 上游 Triton | triton-ascend（本仓库） |
|---|---|---|
| `@triton.jit` 定义在 runtime/jit.py | ✅ 正确 | ✅ 一致 |
| 首次调用触发 JIT 编译 | ✅ 正确 | ✅ 一致 |
| inspect/ast 解析源码 | ✅ 正确 | ✅ 一致 |
| DependenciesFinder（NodeVisitor、依赖分析、hash） | ✅ 正确 | ✅ 一致（jit.py#L34） |
| 文件 `compiler/ast.py` | ❌ 资料未显示存在；解析在 runtime/jit.py | ❌ 不存在 |
| 文件 `compiler/code_generator.py`（AST→TTIR） | ✅ 存在 | ✅ 存在（一致） |
| "TTIR → TTGIR" 阶段 | ✅ 正确（NVIDIA 后端必经） | ❌ 本仓库无 TTGIR |
| 文件 `compiler/lower.py` | ❌ 未见此文件（降级在 C++ pass / 后端 add_stages） | ❌ 不存在 |
| 文件 `compiler/ir.py` | ❌ 未见（类型在 `_C.libtriton.ir` + language/core.py） | ❌ 不存在（同左） |

---

## 7. 关键文件清单

| 职责 | 实际文件 |
|---|---|
| 取源码 + `ast.parse`（AST 入口） | [runtime/jit.py](file:///c:/main/triton-ascend/python/triton/runtime/jit.py)（`JITCallable`，L455-L528） |
| 依赖分析 / 哈希（`DependenciesFinder`） | [runtime/jit.py#L34](file:///c:/main/triton-ascend/python/triton/runtime/jit.py#L34) |
| AST → TTIR（`CodeGenerator` / `ast_to_ttir`） | [compiler/code_generator.py](file:///c:/main/triton-ascend/python/triton/compiler/code_generator.py) |
| 编译调度 / 缓存 / 后端阶段编排（`compile`、`ASTSource`） | [compiler/compiler.py](file:///c:/main/triton-ascend/python/triton/compiler/compiler.py) |
| TTIR → TTAdapter(Linalg) → NPUBin 各阶段 | [third_party/ascend/backend/compiler.py](file:///c:/main/triton-ascend/third_party/ascend/backend/compiler.py) |
| launch 函数生成 | [compiler/make_launcher.py](file:///c:/main/triton-ascend/python/triton/compiler/make_launcher.py) |
| IR / 语言类型 | C++ 绑定 `_C.libtriton.ir` + [language/core.py](file:///c:/main/triton-ascend/python/triton/language/core.py) |

---

## 8. 总结

1. **前端部分**（`runtime/jit.py` + `compiler/code_generator.py`）在两种架构下基本一致：源码 → AST → 依赖分析 → TTIR。
2. **中后端部分**差异明显：上游走 `TTGIR → LLVM IR → PTX/SASS`；triton-ascend 走 `TTAdapter(Linalg) → NPUBin`，没有 TTGIR。
3. **易错点**：`ast.py`、`lower.py`、`ir.py` 三个文件在本仓库均不存在；描述时务必按上面"关键文件清单"引用。
