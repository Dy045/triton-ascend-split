# DynamicCVPipeline 门禁设计文档

## 1. 触发条件

- **自动触发**：PR 修改以下目录时，由 `dynamic-cv-pipeline-trigger.yml` 信号 workflow 完成
  后，`dynamic-cv-pipeline-tests.yml` 通过 `workflow_run` 自动启动
  - `third_party/ascend/lib/DynamicCVPipeline/**`
  - `third_party/ascend/include/DynamicCVPipeline/**`
- **手动触发**：`workflow_dispatch`，可在 Actions 页面手动启动（需提供 `pr_id`）

## 2. 整体流程

采用与 Ascend950 相同的两阶段架构，确保 fork PR 也能安全访问 secrets：

```
PR 提交/更新 (匹配 DynamicCVPipeline 路径)
        │
        ▼
┌──────────────────────────────────────────────┐
│  Workflow 1: DynamicCVPipeline Trigger       │
│  (pull_request, paths 过滤)                  │
│                                              │
│  轻量信号 — 仅标记 "路径变更，需跑远端测试"   │
│  cancel-in-progress: true                    │
└──────────────────────────────────────────────┘
        │ workflow_run (completed)
        ▼
┌──────────────────────────────────────────────┐
│  Workflow 2: DynamicCVPipeline Tests         │
│  (workflow_run / workflow_dispatch)          │
│                                              │
│  在 base-repo 上下文运行，secrets 可用        │
│  1. Resolve PR context (PR 号、head SHA)     │
│  2. Set commit status (pending)              │
│  3. SSH 到远端服务器                         │
│     docker start + exec python ci.py         │
│  4. scp hello.txt 回 runner                  │
│  5. Upload artifact                          │
│  6. Set commit status (final)                │
│  cancel-in-progress: false (串行排队)        │
└──────────────────────────────────────────────┘
```

## 3. 远端服务器

| 项 | 值 |
|---|---|
| IP | 61.47.16.82 |
| 用户 | z00896713 |
| 环境 | Docker 容器 `z00896713` |
| 工作目录 | `/home/z00896713/ci` |
| 入口脚本 | `python ci.py` |
| 产物 | `hello.txt` |

密码通过 GitHub Secret `DYNAMIC_CV_TEST_PASSWORD` 注入，**绝不硬编码在代码或文档中**。

## 4. ci.py 说明

`ci.py` 是远端 Docker 容器内的测试入口脚本，位于 `/home/z00896713/ci/ci.py`。

- **功能**：拉取 PR 代码，分别对 base 和 head 运行 DynamicCVPipeline 相关测试，生成对比结果
- **输入**：通过环境变量或参数获取 PR 信息（PR number、base commit、head commit）
- **输出**：`/home/z00896713/ci/hello.txt`

## 5. 产物查看

每个 PR 运行后会生成一个 artifact，名称为 `hello-txt-pr-<PR_NUMBER>`。查看方式：

1. 进入 PR 页面
2. 点击 CI 完成的 workflow run
3. 在 Summary 页面底部下载 artifact

如果同一个 PR 多次触发，后来的 run 会覆盖之前的 artifact（同名）。

## 6. 并发策略

- **Trigger workflow**: `cancel-in-progress: true`（新 commit 取消旧信号）
- **Tests workflow**: `cancel-in-progress: false`（串行排队，远端资源有限，不中断正在跑的 SSH 会话）

手动 `workflow_dispatch` 触发时使用 `github.run_id` 作为单独的 concurrency group，不跟 PR 的排队冲突。

## 7. 超时配置

Job 设置 `timeout-minutes: 60` 防止远端卡死浪费资源。

## 8. 错误处理

| 场景 | 行为 |
|---|---|
| SSH 连接失败 | Job 直接 fail，commit status 标记 failure |
| docker start 失败 | 远端命令返回非 0，Job fail |
| python ci.py 报错 | docker exec 返回非 0，Job fail |
| hello.txt 不存在 | docker cp + scp 失败，Job fail |

即 **ci.py 失败则门禁不通过**，PR 上会显示红 ✗。不做重试，不区分错误类型。

## 9. 产物保留

GitHub artifact 默认保留 **90 天**，到期自动清理。如需更长时间可在 workflow 中设置 `retention-days`。

## 10. GitHub Secrets 配置

需在 GitHub repo → Settings → Secrets and variables → Actions 中配置：

| Secret 名 | 说明 |
|---|---|
| `DYNAMIC_CV_TEST_HOST` | 远端服务器 IP |
| `DYNAMIC_CV_TEST_USER` | SSH 用户名 |
| `DYNAMIC_CV_TEST_PASSWORD` | SSH 密码 |

## 11. 安全设计

- 使用 `workflow_run` + `workflow_dispatch` 触发，所有 job 在 base-repo 上下文运行
- **绝不** checkout PR head SHA（防止 fork 代码在 secret-bearing token 下执行）
- Secrets 仅在 `workflow_run` 上下文中可访问，fork PR 的 `pull_request` 事件无法获取
- 参考 Ascend950-pipeline-tests.yml 的安全模式

## 12. Workflow 文件

| 文件 | 用途 |
|---|---|
| `.github/workflows/dynamic-cv-pipeline-trigger.yml` | 轻量信号 workflow (pull_request, paths) |
| `.github/workflows/dynamic-cv-pipeline-tests.yml` | 实际测试 workflow (workflow_run / workflow_dispatch) |

## 13. Agent

路径：`.claude/agents/ci-cd.md`（已创建）
