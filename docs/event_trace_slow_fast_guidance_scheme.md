# 基于求解轨迹蒸馏的快慢协同 CDCL-SAT 引导方案

## 1. 方案定位

建议论文主线为：

**Event-Trace Distilled Slow-Fast Guidance for CDCL SAT Solving**

中文表述：

**基于求解轨迹蒸馏的快慢协同 CDCL-SAT 引导方法**

核心思想：

```text
一次 GNN 理解静态 CNF 结构
+ CDCL solver 短 rollout 产生真实搜索事件
+ 轻量 event adapter 多次快速更新变量权重和极性
+ solver 使用更新 guidance 继续求解
```

这不是简单的“多轮 GNN refinement”。最终主方法应是：

```text
Slow model: GNN, 低频或一次调用，用于结构理解
Fast model: Event Adapter, 高频/多轮调用，用于根据 solver event 快速纠偏
CDCL solver: Glucose/Kissat, 接收动态 weight / polarity guidance
```

## 2. 研究动机

### 2.1 RLAF 的基础与局限

RLAF 提出 one-shot SAT solver guidance：

```text
CNF -> Literal-Clause Graph -> GNN -> variable weight + polarity -> solver
```

它的优点是 GNN 只调用一次，推理开销低；缺点是 guidance 是静态的，无法根据搜索过程深处的冲突、传播、学习子句等信息动态调整。

本项目已有代码和论文工作已经从 RLAF 扩展到：

```text
GNN -> warmup solver -> feedback state -> GNN refinement -> final solver
```

但多轮 GNN refinement 仍有两个问题：

1. GNN 调用次数难以确定，且每次 GPU inference 都应计入总时间。
2. 反复调用完整 GNN 代价较高，不适合作为 solver 搜索过程中的频繁反馈机制。

### 2.2 标准 message-passing GNN 的对称性问题

标准 message-passing GNN 的表达能力受 1-WL / color refinement 限制。在高度对称的 CNF 图上，静态 GNN 很难区分结构等价的变量。

需要注意：本方案不声称理论上突破 1-WL。更严谨的表述是：

> CDCL rollout 产生的真实搜索事件为变量引入静态图之外的动态身份，从经验上缓解静态 GNN 对对称变量不可分的问题。

这就是：

**Search-induced symmetry breaking**

即搜索诱导的对称性破缺。

### 2.3 为什么需要轻量 adapter

Graph-Q-SAT 等逐分支调用神经网络的方法容易被推理开销拖垮。RLAF 和 NeuroBack 都强调低频/一次 GNN 调用的重要性。

因此，本方案采用：

```text
GNN: 慢模型，负责全局结构先验
Adapter: 快模型，负责根据事件状态动态调整 guidance
```

这同时解决：

- 静态 GNN guidance 不能适应搜索过程的问题；
- 高频调用完整 GNN 过慢的问题；
- solver runtime reward 稀疏、直接 RL 难训练的问题。

## 3. 最终架构

整体流程：

```text
Input CNF
  |
  v
Literal-Clause Graph
  |
  v
Slow GNN Encoder
  -> variable embedding h_x
  -> initial log weight log w_0(x)
  -> initial polarity logit p_0(x)

Round t:
  1. solver 使用当前 guidance 跑一个 rollout
  2. 收集 event e_t(x) 和 global rollout state g_t
  3. EventEncoder + EMA 得到 memory state s_t(x)
  4. Fast Adapter 输出 residual update
  5. 更新变量 weight / polarity
  6. 进入下一轮 rollout 或 final solve
```

公式化描述：

```text
s_t(x) = lambda * s_{t-1}(x)
       + (1 - lambda) * EventEncoder(e_t(x))

[delta_log_weight_t(x), delta_polarity_logit_t(x)]
  = Adapter(h_x, s_t(x), g_t)

log w_t(x) = clip(log w_0(x) + delta_log_weight_t(x))

polarity_logit_t(x)
  = p_0(x) + delta_polarity_logit_t(x)
```

实际传给 solver 时：

```text
weight_t(x) = exp(log w_t(x))
polarity_t(x) = sigmoid(polarity_logit_t(x)) > 0.5
```

## 4. 模块设计

### 4.1 Slow GNN

GNN 仍使用 literal-clause bipartite graph。

输入：

- literal node features
- clause node features
- 可选 global feedback state
- 可选 variable event state

输出：

```text
h_x: 每个变量的结构 embedding
y_0(x): 初始 guidance logits
```

其中 `y_0(x)` 通常包含：

```text
weight log-mean / log-weight parameter
polarity logit
optional sigma parameter
```

### 4.2 Event State

第一阶段建议只做 variable-level / literal-level event，不主推 clause-level 动态图。

当前可用事件：

```text
event_var_decisions
event_var_propagations
event_var_conflict_lits
event_var_learnt_lits
event_var_activity
```

其中语义应严格区分：

- `event_var_decisions`: 变量被作为 branching decision 的次数。
- `event_var_propagations`: 变量通过 unit propagation 被赋值的次数。
- `event_var_conflict_lits`: conflict analysis 中触达/参与冲突图分析的变量次数。
- `event_var_learnt_lits`: 最终 learned clause 中出现的变量次数。
- `event_var_activity`: solver 当前变量 activity。

建议使用增强特征，不只用累计 `log1p`：

```text
cumulative log features:
  log1p(decision_count)
  log1p(propagation_count)
  log1p(conflict_involvement)
  log1p(learnt_clause_involvement)
  log1p(activity)

delta/window features:
  当前 rollout 窗口内的同类计数

rate features:
  delta_count / total_delta_count

rank features:
  within-instance percentile / rank
```

当前代码中的 enhanced event state 是 20 维：

```text
5 cumulative log
5 delta log
5 rate
5 rank
```

后续如果 C++ 侧继续增强，可加入：

```text
last_decision_level
last_polarity
assigned_ratio
pos/neg propagation count
pos/neg conflict count
low-LBD learnt involvement
UIP involvement
```

### 4.3 Event Memory / EMA

单轮 event 可能有噪声，尤其是 solver 陷入局部搜索区域时，短窗口事件会有偏。

因此引入 EMA：

```text
s_t = momentum * s_{t-1} + (1 - momentum) * encoded_event_t
```

建议默认：

```yaml
state_momentum: 0.5
```

可在实验中消融：

```text
momentum = 0.0 / 0.5 / 0.7 / 0.9
```

### 4.4 Fast Event Adapter

Adapter 是一个轻量 per-variable MLP。

输入：

```text
concat(
  h_x,                 # GNN 静态结构 embedding
  s_t(x),              # event memory
  g_t                  # global rollout state, optional
)
```

输出：

```text
delta_log_weight
delta_polarity_logit
```

使用 residual update，而不是直接覆盖 GNN 输出：

```text
new_guidance = base_guidance + adapter_delta
```

优点：

- 保留 GNN 的静态结构先验；
- adapter 初始为零残差时等价于 RLAF / base GNN；
- 训练更稳定；
- 方便从 RLAF checkpoint 初始化。

### 4.5 Rollout Budget

推荐优先使用 conflict budget，而不是 CPU time budget。

原因：

- conflict count 更可复现；
- 不容易受机器负载影响；
- 更符合 CDCL 搜索过程的单位。

建议初始配置：

```yaml
rollout_budget_type: conflicts
rollout_conflicts: 500
warmup_cpu_lim: 1   # 作为安全 guard
num_rounds: 2
```

最终论文仍必须以 wall-clock total time 为主指标。

## 5. 训练范式

建议三阶段训练：

### 5.1 Stage 1: RLAF 初始化

先训练或加载 one-shot RLAF GNN。

目标：

```text
CNF graph -> initial weight / polarity
```

这个阶段提供静态结构先验。

### 5.2 Stage 2: Trace Distillation 预训练 Adapter

直接用最终 solver runtime 做 RL 太稀疏，因此需要从 solver trace 中提取更密集的伪标签。

候选伪标签：

```text
low-LBD learnt clause involvement
useful decision count
conflict analysis involvement
propagation impact
UIP involvement, optional
```

建议第一版 soft target：

```text
y_x =
  a * rank(low_lbd_learnt_count)
+ b * rank(useful_decision_count)
+ c * rank(conflict_involvement)
+ d * rank(propagation_impact)
```

如果暂时没有 LBD / UIP 事件，可以先用现有事件：

```text
y_x =
  a * rank(event_var_learnt_lits)
+ b * rank(event_var_conflict_lits)
+ c * rank(event_var_decisions)
+ d * rank(event_var_propagations)
```

Adapter 预训练目标：

```text
让 adapter 上调高价值变量的 log_weight
让 adapter 学习更好的 polarity residual
```

### 5.3 Stage 3: Solver-in-the-loop 微调

在预训练基础上，用 solver-in-loop GRPO 或 preference optimization 微调。

奖励建议：

```text
R = -(
  alpha * log1p(total_time)
+ beta  * log1p(conflicts)
+ gamma * log1p(decisions)
)
```

注意：

- `total_time` 必须包含 GNN、adapter、rollout、final solve。
- 训练和评估协议要一致。
- 如果评估使用 2 rounds，训练最好也见过相似状态。

## 6. 推理协议

推荐默认推理：

```text
Round 0:
  GNN(graph) -> base guidance

Rollout 1:
  solver(base guidance), budget = 500 conflicts
  collect event state

Adapter 1:
  update guidance

Rollout 2:
  solver(updated guidance), budget = 500 conflicts
  collect event state

Adapter 2:
  update guidance

Final:
  solver(final guidance), full budget
```

当前工程实现是重启式 refinement：

```text
每轮 rollout 是一次短 solver run
收集事件后重新生成 guidance
final solver 使用最终 guidance 从头求解
```

论文中必须明确这一点：

- 当前不是 pause/resume 同一个 CDCL solver state；
- learnt clauses/trail 不跨轮保留；
- event state 是从前序 rollout 中抽取的引导信号。

未来可以探索 resume-style solver integration，但第一篇不建议把这作为主线。

## 7. 实验设计

### 7.1 主基线

必须包含：

```text
Glucose
Kissat, 如果工程时间允许
RLAF one-shot
global feedback refinement
event-var multi-round GNN refinement
one-shot GNN + event adapter
trace-pretrained event adapter
trace-pretrained + GRPO event adapter
```

### 7.2 控制组 / 消融

关键消融：

```text
shuffled event state
random event state
adapter without GNN embedding
adapter without EMA memory
adapter without trace distillation
activity-only
decision-only
propagation-only
conflict-only
learnt-only
1 / 2 / 3 rollout rounds
rollout_conflicts = 100 / 500 / 1000 / 5000
```

`shuffled event state` 很重要，用来证明增益来自真实事件和变量结构的对应关系，而不是噪声或额外参数量。

### 7.3 对称性压力测试

为了支撑 search-induced symmetry breaking，建议加入：

```text
variable permutation stress test
graph coloring
regular graph coloring
pigeonhole
Tseitin formulas
```

实验问题：

1. 静态 GNN 是否给对称变量相同或高度相似的 guidance？
2. solver rollout 后 event state 是否能区分这些变量？
3. adapter 是否利用这种动态差异提升求解效率？
4. 随机变量重命名后方法是否稳定？

### 7.4 指标

SAT 论文中不能只报平均时间。

建议报告：

```text
mean wall-clock time
median wall-clock time
PAR-2
solved count
SAT / UNSAT split
cactus plot
GNN time
adapter time
rollout solver time
final solver time
total time
```

其中 `total time` 是主指标。

## 8. 当前代码状态

当前项目中已经落地的部分：

### 8.1 Enhanced event state

文件：

```text
src/solving/state.py
```

能力：

```text
legacy 5-dim event state
enhanced 20-dim event state
EMA event_memory
```

### 8.2 Slow-fast event adapter

文件：

```text
src/model/model.py
src/policy/evaluate.py
```

能力：

```text
GNN 首轮返回 base_embedding 和 base_y
sample_var_params 可缓存这些变量级特征
后续轮次存在 event_state 时走 event_adapter 快路径
adapter 使用 zero-initialized residual
```

### 8.3 严格总时间统计

文件：

```text
evaluate_guided_solver.py
```

能力：

```text
final solver CPU time
final GNN / adapter time
refinement rollout CPU time
refinement guidance GPU/CPU time
total time
```

### 8.4 Conflict-budget rollout

文件：

```text
src/solving/budget.py
solvers/glucose_weighted/simp/Main.cc
```

能力：

```text
rollout_budget_type: conflicts
rollout_conflicts: 500
glucose_weighted 支持 -conf-lim
```

### 8.5 solver event 语义修正

文件：

```text
solvers/glucose_weighted/core/Solver.cc
```

当前语义：

```text
event_var_conflict_lits: conflict analysis 触达变量
event_var_learnt_lits: 最终 learnt clause 中出现的变量
```

### 8.6 适配 RLAF checkpoint 初始化

文件：

```text
train_rlaf.py
```

能力：

```text
按当前 config 构建 adapter 模型
从旧 checkpoint 只加载形状匹配的 tensor
跳过 adapter 新增参数和维度不匹配参数
```

这支持：

```text
RLAF one-shot checkpoint -> adapter model warm start
```

## 9. 推荐配置

训练 event adapter：

```yaml
defaults:
  - config_train_rlaf_balanced
  - _self_

model_name: GNN_Glucose_3SAT_EventVarFeedback

training:
  feedback_state_type: event_var
  feedback_event_state_features: enhanced
  feedback_state_momentum: 0.5
  feedback_input_mode: mixed_warmup
  feedback_val_mode: model_warmup
  feedback_warmup_budget_type: conflicts
  feedback_warmup_conflicts: 500

model:
  global_state_dim: 0
  var_state_dim: 20
  event_adapter:
    enabled: true
    hidden_dim: 128
```

评估：

```yaml
feedback_refinement:
  enabled: true
  state_type: event_var
  event_state_features: enhanced
  num_rounds: 2
  state_momentum: 0.5
  rollout_budget_type: conflicts
  rollout_conflicts: 500
  warmup_cpu_lim: 1
```

## 10. 论文写法建议

### 10.1 推荐摘要表述

可以这样写：

> Existing neural SAT guidance methods are either static, relying on one-shot predictions before search, or too expensive, requiring frequent neural inference during branching. We propose an event-trace distilled slow-fast guidance framework for CDCL SAT solving. A slow GNN extracts static structural priors from the CNF graph once, while a lightweight event adapter repeatedly updates variable weights and polarities from real CDCL rollout events. These events provide search-induced dynamic identities that empirically mitigate the inability of static message-passing GNNs to distinguish symmetric variables. The adapter is first distilled from solver traces and then fine-tuned with solver-in-the-loop feedback under strict total-time accounting.

### 10.2 贡献点写法

建议写成四点：

1. 提出 search-induced dynamic state，用真实 CDCL event 为变量提供静态 CNF 图之外的动态身份。
2. 提出 slow-fast neural guidance，用一次 GNN 和多轮轻量 adapter 实现低开销闭环引导。
3. 提出 trace-distilled event adapter，将 solver trajectory 中的低 LBD、冲突、传播、决策信号转化为密集训练监督。
4. 在严格 total-time accounting 下系统比较 one-shot、multi-round GNN refinement 和 event adapter。

### 10.3 避免过度声明

不要写：

```text
We solve the 1-WL limitation of GNNs.
```

建议写：

```text
We use solver interaction to introduce dynamic evidence beyond the static CNF graph, empirically mitigating symmetry-induced indistinguishability in message-passing GNN guidance.
```

## 11. 相关文献

### RLAF

Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs  
https://arxiv.org/abs/2505.16053

支撑点：

- one-shot GNN guidance；
- variable weight + polarity；
- GRPO with solver feedback；
- 你的直接 baseline。

### NeuroBack

NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks  
https://proceedings.iclr.cc/paper_files/paper/2024/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html

支撑点：

- 避免频繁在线 GNN 调用；
- GNN guidance 接入 CDCL solver；
- backbone/polarity guidance。

### NeuroCore

Guiding High-Performance SAT Solvers with Unsat-Core Predictions  
https://arxiv.org/abs/1903.04671

支撑点：

- 变量级预测可以进入 VSIDS/activity 机制；
- unsat-core prediction 作为 solver guidance。

### Graph-Q-SAT

Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?  
https://proceedings.neurips.cc/paper/2020/hash/6d70cb65d15211726dcce4c0e971e21c-Abstract.html

支撑点：

- RL + GNN branching；
- 高频 neural inference 容易成为瓶颈；
- 支撑本方案避免 per-branching GNN。

### RDC-SAT

Learning Splitting Heuristics in Divide-and-Conquer SAT Solvers with Reinforcement Learning  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/f5c683b93319b82689af3afc71257df2-Abstract-Conference.html

支撑点：

- 使用 learned clauses、activity、LBD 等动态 solver state；
- 说明 CDCL 内部状态是有价值的学习信号。

### ImitSAT

Boolean Satisfiability via Imitation Learning  
https://openreview.net/forum?id=LNqWbY5iIf

支撑点：

- solver trace / KeyTrace 可作为密集监督；
- 支撑 trace distillation。

### NeuroSelect

NeuroSelect: Learning to Select Clauses in SAT Solvers  
https://www.cse.cuhk.edu.hk/~byu/papers/C228-DAC2024-NeuroSelect.pdf

支撑点：

- clause-level CDCL 信号重要；
- learned clause / deletion policy 可学习；
- 也说明 clause-level 动态图工程复杂，适合后续工作。

### GNN 表达力限制

How Powerful are Graph Neural Networks?  
https://arxiv.org/abs/1810.00826

Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks  
https://ojs.aaai.org/index.php/AAAI/article/view/4384

支撑点：

- message-passing GNN 与 1-WL / color refinement 的关系；
- 支撑 search-induced symmetry breaking 的理论动机。

## 12. 下一步工作

建议按优先级推进：

1. **实现 trace pseudo-label 生成**
   - 从 solver event 中生成变量级 soft target。
   - 第一版使用 existing events：learnt/conflict/decision/propagation rank。
   - 后续加入 low-LBD 和 UIP。

2. **实现 adapter distillation training**
   - 先冻结或半冻结 GNN；
   - 训练 adapter 输出 residual；
   - 验证 adapter 是否学会提升高价值变量权重。

3. **做小规模验证**
   - 3SAT-300 / 350；
   - 对比 one-shot、global feedback、event-var GNN refinement、adapter。

4. **补严格实验协议**
   - cactus plot；
   - PAR-2；
   - SAT/UNSAT split；
   - shuffled event state；
   - variable permutation stress test。

5. **考虑接 Kissat**
   - 如果目标是顶会，只有 Glucose 可能显得基线偏弱；
   - Kissat 可作为强 solver baseline。

## 13. 总结

最终方案不是“把 GNN 做得更大”，也不是“每次 branching 都调用神经网络”。

核心路线是：

```text
让 GNN 做一次结构理解，
让 CDCL 搜索事件打破静态对称性，
再让轻量 adapter 把事件反馈转化为动态 guidance。
```

这条路线同时回应：

- RLAF one-shot static guidance 不够动态；
- 标准 GNN 对称性表达有限；
- 高频神经 branching 推理太慢；
- solver runtime reward 太稀疏。
