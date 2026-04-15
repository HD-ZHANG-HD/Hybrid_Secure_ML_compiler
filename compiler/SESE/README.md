# SESE Compiler Plan

## 1. Goal

本目录用于实现一个新的 compiler 方法 `SESE`。

它解决的问题不是单纯的 HE/MPC 二值划分，而是对整个 computation graph 联合决定：

- 每个 operator 在 `HE` 还是 `MPC`
- 何时插入 `HE -> MPC` conversion
- 何时插入 `MPC -> HE` conversion
- 何时插入 `bootstrap`
- 在 `HE` 执行过程中如何满足 level / noise budget 约束

目标函数保持为 `latency-only`：

- `C(P) = sum_e c(e)`

与已有 `state_expanded_opt` 的区别在于：

- 链式子图仍然可以做严格的状态扩展最短路
- 对一般 DAG，尤其 residual / shortcut / merge，不再直接把联合状态暴露到全局
- 我们通过图拓扑分析，把状态爆炸严格封锁在局部 `SESE`（single-entry single-exit）区域内
- 对外只暴露一个常数大小的块摘要，因此全局优化只在块级图上进行

## 2. Core Idea

### 2.1 状态图只在块内展开

全局原始状态可以写为：

- `s = (i, d, l)`
- `i`: 即将执行的拓扑位置
- `d in {HE, MPC}`
- `l in {0, 1, ..., L}`，仅在 `d = HE` 时有效

如果直接把一般 DAG 上 merge 节点的多路联合状态写成：

- `(v, sigma_1, sigma_2, ..., sigma_k)`

则状态空间会随分支数迅速膨胀。`SESE` 方法的核心是：

- 先将 DAG 分解为若干单入口单出口区域
- 对每个区域内部做严格状态扩展和 merge 对齐分析
- 再把整个区域压缩成一个常数大小的 transfer summary

因此，复杂联合状态只存在于区域内部，不泄漏到区域外部。

### 2.2 对外暴露 O(1) 状态转移

对任意一个 SESE 区域 `B`，定义边界状态集合：

- `Sigma = {(HE, 0), (HE, 1), ..., (HE, L), MPC}`

其大小为：

- `|Sigma| = L + 2`

当 `L` 被视为系统配置常数时，`|Sigma| = O(1)`。

因此每个区域对外只暴露一个常数大小摘要：

- `Summary_B[s_in][s_out] = min latency`

表示：

- 从入口边界状态 `s_in`
- 经过块内所有合法动作
- 到出口边界状态 `s_out`
- 的最小代价

可选地，摘要中还保留：

- 最优块内 action trace
- 块内 conversion / bootstrap 插入位置
- merge 对齐说明

全局层面只需要在 SESE 摘要图上组合这些块摘要。

## 3. Graph Decomposition Plan

### 3.1 预处理

先对 `OperatorGraph` 建立增强视图：

- `pred(v)`, `succ(v)`
- topological order
- synthetic `ENTRY`
- synthetic `EXIT`

其中：

- 所有 source node 从 `ENTRY` 引出
- 所有 sink node 指向 `EXIT`

### 3.2 SESE 区域识别

在 DAG 上做：

- dominator analysis
- post-dominator analysis

一个候选区域 `B = (u, v)` 满足：

- `u` dominates `v`
- `v` post-dominates `u`

并且该区域内部节点集合定义为：

- 所有被 `u` 支配且被 `v` 后支配的节点

我们优先识别两类区域：

1. 原子链段
2. residual / shortcut / merge 包围的最小闭合区域

这样可以形成一个 region tree：

- 叶子是原子 operator 或短链
- 内部节点是更大的 SESE block

### 3.3 为什么这样能控制状态膨胀

因为 residual 的多路输入对齐问题只会出现在某个 merge 所在的最小闭合 SESE 区域内：

- 分支从同一个 entry 分开
- 最终在同一个 exit 闭合

所以联合状态无需跨越整个 DAG 传播，只需在该闭合区域内部求解一次。

## 4. Local Exact Solver Inside One SESE Block

### 4.1 块内输入输出接口

对 SESE block `B`：

- 输入：一个入口边界状态 `s_in in Sigma`
- 输出：一个出口边界状态 `s_out in Sigma`

块内求解的本质是：

- 给定 `s_in`
- 枚举块内合法动作
- 找到到达每个 `s_out` 的最小代价

于是得到一个 transfer matrix。

### 4.2 块内状态

块内状态不再只记录单个节点的 `(domain, level)`，还要记录 merge 对齐所需的局部信息。

建议使用两层表示：

1. 执行态 `ExecState`
   - 当前已完成的块内 frontier
   - 每个 frontier tensor 的 `(domain, level)`
2. 规范化态 `CanonicalState`
   - 对同构 frontier 排序并归一化
   - 用于去重和最短路 / DP

对链式块，`ExecState` 会退化为已有的 `(i, d, l)`。

### 4.3 合法动作

块内动作沿用你的定义：

1. `execute_he`
2. `execute_mpc`
3. `convert_he_to_mpc`
4. `convert_mpc_to_he`
5. `bootstrap_he`

但在 merge 节点上增加一个显式约束：

- 所有输入必须在目标方法要求的 domain 下对齐
- 若输入不对齐，则动作集合中必须先包含必要的 conversion
- 若目标是 HE merge，还必须满足预算可行性

### 4.4 merge compatibility

对一个 `k` 输入 merge 节点，定义：

- `compat(v, sigma_1, ..., sigma_k, target_domain)`

它返回：

- 是否可行
- 对齐后输入状态
- 为实现对齐所需的最小 conversion 代价

关键原则：

- merge 的“不一致代价”只在块内求一次
- 块外不再跟踪每条支路的独立状态

### 4.5 块内求解算法

建议分两种情况：

1. `SESE-chain`
   - 直接复用现有 `state_expanded_opt` 的 exact shortest path
2. `SESE-branch`
   - 在块内拓扑序上做 label-setting shortest path / DP
   - 状态键为规范化后的 frontier state

局部最短路输出：

- `best_cost[s_in][s_out]`
- `best_trace[s_in][s_out]`

## 5. Global Composition After Compression

### 5.1 块级图

将原图压缩为 block graph：

- 节点是 atomic op 或 SESE block
- 边表示块之间的数据流

对每个块，只保留常数大小接口 `Summary_B`。

### 5.2 全局 DP / 最短路

在块级拓扑序上做动态规划：

- 输入状态仍然只取自 `Sigma`
- 每经过一个块，就应用其 transfer summary

这一步不再看到块内多路联合状态，因此复杂度与块数线性相关，而不与块内分支结构直接耦合。

### 5.3 计划回放

全局最优路径确定后：

- 沿 block graph 回溯所选 `s_in -> s_out`
- 读取对应块内 `best_trace`
- 拼接成完整 execution plan

最终输出与现有 runtime adapter 风格对齐：

- operator steps
- conversion steps
- bootstrap steps

## 6. Proposed Module Layout

本目录建议按下面的模块边界逐步落地：

- `README.md`
  - 方法说明与设计文档
- `region_analysis.py`
  - dominator / post-dominator
  - synthetic entry/exit
  - SESE region extraction
  - region tree construction
- `region_types.py`
  - `BoundaryState`
  - `SESEBlock`
  - `BlockSummary`
- `local_state.py`
  - 块内执行态与规范化态
- `merge_model.py`
  - merge compatibility / alignment cost
- `local_solver.py`
  - 单个 SESE block 的精确求解
- `summary_builder.py`
  - 生成 `Summary_B[s_in][s_out]`
- `global_solver.py`
  - 在块级摘要图上做全局 DP / 最短路
- `plan_builder.py`
  - 从 block trace 恢复完整 compiler plan
- `runtime_plan_adapter.py`
  - 转换为 runtime `ExecutionPlan`
- `demo.py`
  - 使用测试图跑通 end-to-end
- `test/`
  - 链式 / residual / nested residual / invalid budget case

## 7. Relationship To Existing Code

### 7.1 直接复用

建议尽量复用现有代码：

- `compiler.state_expanded_opt.cost_model.StateExpandedCostModel`
- `compiler.capability_checker`
- `compiler.state_expanded_opt.graph_model.GraphView`
- `compiler.min_cut.runtime_plan_adapter.resolve_method_name`

### 7.2 需要替换的地方

当前 `state_expanded_opt.solve_state_expanded()` 中：

- `chain_exact` 是精确的
- `stage_local_dag` 是 stage 级局部近似

`SESE` 的目标是替换后者的 DAG 处理思想：

- 不再以单节点 stage 贪心处理 merge
- 改成以 SESE block 为单位做局部精确求解

## 8. Complexity Story

设：

- `n` = 原图节点数
- `b` = SESE block 数
- `K = |Sigma| = L + 2`

则：

- 全局层复杂度约为 `O(b * K^2)` 或同量级
- 每个块的复杂度由其内部结构决定
- 但块内复杂状态不会扩散到全局

因此该方法的关键收益不是“完全消除状态爆炸”，而是：

- 将状态爆炸限制在局部
- 用拓扑分解把难问题隔离进少数 residual / merge block
- 让全局组合只面对常数大小的边界状态

## 9. Implementation Milestones

### Milestone 1: Skeleton

- 建立 `compiler/SESE/` 目录
- 定义 region/block/summary 数据结构
- 复用现有 cost model

### Milestone 2: Region Analysis

- 加入 synthetic entry/exit
- 实现 dominator / post-dominator
- 提取最小 residual SESE block

### Milestone 3: Local Solver

- 先支持 `SESE-chain`
- 再支持单个 residual merge block
- 输出 transfer summary

### Milestone 4: Global Composition

- 构建 block graph
- 做摘要级 DP / shortest path
- 回放块内 trace

### Milestone 5: Runtime Integration

- 产出与现有 runtime plan 兼容的 step 列表
- 对比 `state_expanded_opt` 在链式图上的结果一致性

## 10. First Concrete Deliverable

第一阶段不直接写完整求解器，而是先完成以下可验证目标：

1. 对输入 `OperatorGraph` 做拓扑分析并提取 SESE block
2. 为每个 block 输出：
   - `entry`
   - `exit`
   - 内部节点
   - block 类型：`chain` / `residual` / `generic`
3. 为每个 block 预留常数大小边界状态接口：
   - `BoundaryState = (HE, l)` or `MPC`
   - `BlockSummary[K][K]`

这一步完成后，就已经把“状态爆炸被局部封锁”的编译骨架搭好了。

## 11. Recommended Next Step

建议下一步按最小闭环实现：

1. 先实现 `region_analysis.py`
2. 只识别最小 residual-style SESE block
3. 对 `chain` block 直接复用现有 exact solver
4. 对 `residual` block 先支持两路 merge
5. 在一个 `graph_residual_stage_local.json` 的变体上验证

这样可以最快把概念验证成能跑的 compiler 原型。
