# A Cluster-Level Information Fusion Framework for D-S Evidence Theory with Its Applications in Pattern Classification

`ClusterBBA` 是一个用于多源信息融合的 Python 项目，它是论文 **"A Cluster-Level Information Fusion Framework for D-S
Evidence Theory with its Applications in Pattern Classification"** 的源代码。该框架旨在解决经典 Dempster-Shafer (D-S)
证据理论在处理高度冲突证据时可能产生的违反直觉问题。

传统方法通常进行成对的证据比较，而本框架将分析视角从 **BBA-to-BBA**提升到 **BBAs-to-BBAs**
的整体视角，通过将相似证据分簇来更系统地管理和定位冲突，从而在不确定性推理中提供更可靠、更具可解释性的决策支持。

#### 理论背景与动机

D-S证据理论为处理不确定信息提供了强大的数学工具，但在融合来自多个源的高度冲突证据时，其经典的组合规则（Dempster's Rule of
Combination）可能会导致与直觉相悖的结论。问题的根源在于，简单的冲突加权或丢弃无法有效识别和处理由不可靠信源导致的群体性偏见。

当证据集内部存在共识群体和孤立的冲突证据时，传统的成对比较方法难以捕捉这种宏观结构。例如，在多传感器目标识别任务中，即使多数传感器一致指向目标A，一个或少数几个传感器给出的强冲突证据（如指向目标B）也可能严重污染最终的融合结果。

为此，`ClusterBBA` 提出了一种全新的**簇级分析范式**
。其核心思想是：在融合之前，首先识别证据内部的“意见团体”，将观点相似的证据聚合为“簇”，而将观点相异的证据分离开。这样，原始证据间的冲突就转化为簇与簇之间的差异，从而可以在一个更高、更宏观的层面上进行分析和处理，使得冲突的来源和结构更加清晰，为后续的加权融合提供了更可靠的依据。

#### 核心方法论

本框架的实现分为两个核心阶段：**在线证据聚类**和**基于簇结构的加权融合**。

###### 1. 证据聚类与簇质心构建

为了捕捉每个证据簇的代表性特征，我们引入了基于分形理论的**最大Deng熵分形 (`fractal/fractal_max_entropy.py`)**
。该算子通过迭代过程，以最大化信息熵（即最不引入主观偏见）的方式揭示证据的内在层次结构。

每个簇的**质心**被定义为其内部所有成员证据经过分形变换后的算术平均值 (`cluster/one_cluster.py`)
。簇质心的更新采用了一种高效的递归方式，大大降低了在线聚类过程中的计算复杂度。

###### 2. 簇间散度 $D_{CC}$

为了量化不同证据簇之间的差异，我们设计了一种新颖的**簇间散度度量 $D_{CC}$ (`divergence/rd_ccjs.py`)**
。与传统的散度度量不同，$D_{CC}$ 能够同时捕捉两个关键维度的差异：

- **置信强度**：簇质心在各个命题上的置信度分布差异。
- **结构支持度**：簇内成员对不同命题支持的广泛程度（即一个命题是由簇内多数成员共同支持，还是仅由少数成员支持）。

$D_{CC}$ 满足非负性、对称性和三角不等式等伪度量性质，为衡量簇间分离度提供了坚实的数学基础。

###### 3. 动态证据分配

当一个新的证据到来时，框架采用一种**基于奖励的贪婪分配规则 (`cluster/multi_clusters.py`)**
来决定其归属。系统会评估将该证据分配给每一个现有簇或创建一个新簇的所有可能策略。

每种策略的“奖励”函数被设计为最大化**平均簇间分离度**（由 $D_{CC}$ 衡量）与最小化**平均簇内一致性**（由 Belief
Jensen-Shannon 散度衡量）的比值。通过选择奖励最高的策略，系统能够动态地维护一个既能有效分离冲突观点，又能保持内部观点一致的簇结构。

###### 4. 两阶段信息融合

在所有证据完成聚类后，框架进入第二阶段的加权融合 (`cluster/cluster_weights_calculator.py`)。每个证据的最终可信度权重由三个因素共同决定：

1. **所属簇的大小**：更大的簇通常代表更强的共识。
2. **证据与所属簇的一致性**：证据与簇内其他成员的平均散度。
3. **所属簇与其他簇的分离度**：该簇与其他所有簇的平均`D_CC`散度。

此外，框架引入了**专家偏置系数 $\alpha$**，允许用户根据先验知识调整对大簇（共识）与小簇（少数派意见）的信任程度，从而在不同应用场景下实现更灵活的决策。

#### 实验与验证

本框架在多个UCI基准数据集（如Iris, Wine, Seeds, Glass）上进行了模式分类任务的验证 (`experiments/`)。实验结果表明：

- 与传统的D-S证据理论方法（如Dempster's Rule, Murphy's method, Deng's
  method等）相比，本框架在分类准确率和F1分数上均表现出显著优势，尤其是在处理类别不平衡和特征高度冲突的复杂数据集（如Glass）时，鲁棒性更强。
- 通过消融实验验证了框架中各个创新点（特别是 `D_CC` 度量和专家偏置系数 `α`）的有效性。
- 框架中的关键超参数 `(μ, λ)` 对模型性能有重要影响，我们采用**贝叶斯优化 (`experiments/tune_hyperparams_bayes.py`)**
  的方法进行高效、数据驱动的自动寻优，以适应不同数据集的内在特性。

#### 仓库结构

```
/
├── baseline/               # 用于比较的其他基线方法（如 DS-SVM, Evidential Deep Learning）
├── cluster/                # 实现簇的构建、更新和权重计算
├── data/                   # 包含数据集（Iris, Wine, Seeds, Glass）和BBA生成脚本
├── divergence/             # 实现各种散度度量（BJS, Jousselme, D_CC 等）
├── entropy/                # 实现 Deng 熵和信息量计算
├── experiments/            # 包含论文中的所有实验、应用和超参数调整脚本
├── experiments_result/     # 存储实验生成的图表和结果数据
├── figures/                # 用于生成论文图表的脚本
├── fractal/                # 实现最大Deng熵分形算子
├── fusion/                 # 实现不同的证据融合规则（D-S, Murphy, Deng等）
├── mean/                   # 计算均值BBA和冲突系数
├── utility/                # 提供数据读取、绘图样式和概率转换等辅助功能
├── config.py               # 项目配置文件
├── main.py                 # 主执行文件入口
└── requirements.txt        # 项目依赖
```

#### 安装和使用

1. 克隆本仓库：

   ```
   git clone [https://github.com/your-username/ClusterBBA.git](https://github.com/your-username/ClusterBBA.git)
   cd ClusterBBA
   ```

2. 创建并激活一个虚拟环境（推荐）：

   ```
   python -m venv venv
   source venv/bin/activate  # on Windows use `venv\Scripts\activate`
   ```

3. 安装所需的依赖包：

   ```
   pip install -r requirements.txt
   ```

4. 代码采用了模块化结构。以 `test` 开头的脚本带有主函数，为测试脚本；其他脚本为功能脚本，脚本和脚本之间互为依赖。

#### 贡献

我们欢迎任何形式的贡献，包括 bug 修复、理论扩展或文档改进。请随时提交 Pull Request 或创建 Issue。

#### DOI

```
https://doi.org/10.3390/math13193144
```

#### 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权。
