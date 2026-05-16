# ClusterBBA: A Cluster-Level Information Fusion Framework for D-S Evidence Theory with Its Applications in Pattern Classification

[![Paper](https://img.shields.io/badge/Paper-Mathematics%202025%2C%2013(19)%2C%203144-blue)](https://doi.org/10.3390/math13193144)

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fmath13193144-blue)](https://doi.org/10.3390/math13193144)

[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](#installation)

`ClusterBBA` is the reference implementation for the paper:

> **Ma, M.; Fei, L.** A Cluster-Level Information Fusion Framework for D-S Evidence Theory with Its Applications in Pattern Classification. *Mathematics* **2025**, *13*(19), 3144. https://doi.org/10.3390/math13193144

Instead of assessing evidence only through pairwise BBA-to-BBA comparisons, the framework first organizes similar Basic Belief Assignments (BBAs) into clusters, models consensus and conflict at the group level, and then performs credibility-weighted evidence fusion.

The central idea is simple: when multiple sources provide uncertain or conflicting evidence, the conflict is often not merely an isolated pairwise phenomenon. Similar BBAs may form a coherent evidence group, while unreliable or structurally different BBAs may form separate groups. By explicitly modeling these groups, the proposed framework makes evidential conflict more interpretable and improves the robustness of fusion-based pattern classification.

## Key Features

- **Cluster-level view of evidence conflict**: moves from pairwise evidence comparison to a BBAs-to-BBAs perspective.

- **Maximum-Deng-entropy fractal centroid**: constructs cluster centroids using a maximum-entropy fractal operation over focal elements.

- **Cluster-cluster divergence**: implements a cluster-level divergence measure, denoted as $D_{CC}$ in the paper and historically named `RD_CCJS` in the code.

- **Reward-driven online evidence assignment**: dynamically decides whether a new BBA should join an existing cluster or form a new cluster.

- **Cluster-aware credibility weighting**: computes BBA credibility from cluster size, intra-cluster coherence, and inter-cluster separation.

- **Pattern-classification experiments**: includes BBA generation, classical D-S fusion baselines, machine-learning baselines, Bayesian hyperparameter search, and evaluation scripts.

## Why Cluster-Level Fusion?

Classical Dempster's Rule of Combination can produce counter-intuitive results under severe conflict. Many existing methods mitigate this by modifying the fusion rule or by reweighting BBAs before fusion. However, purely pairwise credibility estimation may miss the structure of the whole evidence set.

`ClusterBBA` addresses this by separating two kinds of information:

1. **Intra-cluster coherence**: whether BBAs inside the same cluster support similar propositions.

2. **Inter-cluster divergence**: whether different clusters represent genuinely different belief structures.

This makes the final decision less dependent on a single highly conflicting BBA and more sensitive to the global structure of evidential consensus.

## Method Overview

The proposed framework consists of two main stages.

#### Stage 1: Sequential Cluster Construction

Incoming BBAs are processed one by one. For each new BBA, the algorithm evaluates all candidate strategies: joining each existing cluster or creating a new cluster. The strategy with the highest reward is selected.

For a cluster $Clus_i$, its fractal order is determined by its size:

$$
h_i = n_i - 1.
$$

The cluster centroid is constructed from the maximum-Deng-entropy fractal BBAs:

$$
\begin{aligned}
\widetilde{m}_{F_i}^{(h)}(A)
&= \frac{1}{n_i}
\sum_{j=1}^{n_i} m_{F_j}^{(h)}(A),
\quad A \subseteq \Theta.
\end{aligned}
$$

The cluster-cluster divergence is computed after aligning centroids to the same global fractal order. In the paper it is denoted by $D_{CC}$:

$$
\begin{aligned}
D_{CC}(Clus_p, Clus_q)
&= \sqrt{
\sum_{A \subseteq \Theta}
\left(
\sqrt{w_p(A)\widehat{m}_{F_p}^{(H)}(A)}
{}-
\sqrt{w_q(A)\widehat{m}_{F_q}^{(H)}(A)}
\right)^2
}.
\end{aligned}
$$

The reward for a candidate assignment strategy is:

$$
\begin{aligned}
R_k
&=
\frac{
\left(
\frac{1}{P_k}
\sum_{1 \leq i < j \leq K_k}
D_{CC}(Clus_i, Clus_j)
\right)^\mu
}{
\left(
\frac{1}{K_k}
\sum_{i=1}^{K_k}
D_{intra}(Clus_i^+)
\right)^\lambda
}.
\end{aligned}
$$

Here, $\mu$ controls the sensitivity to inter-cluster divergence, and $\lambda$ controls the sensitivity to intra-cluster divergence.

#### Stage 2: Cluster-Aware Evidence Fusion

After clustering, each BBA receives a credibility weight. For BBA $m_{i,j}$ in cluster $Clus_i$, the support degree is:

$$
\begin{aligned}
Sup_{i,j}
&= n_i^\alpha
\exp\left(-d_{i,j}^{\lambda}\right)
\exp\left(-D_i^{\mu}\right),
\end{aligned}
$$

where $\alpha$ is an expert-bias coefficient that controls how strongly the algorithm trusts large clusters. The credibility weights are normalized and used to construct a weighted-average BBA:

$$
\begin{aligned}
\bar{m}(A)
&= \sum_i \sum_j Crd_{i,j} m_{i,j}(A).
\end{aligned}
$$

The weighted-average BBA is then fused recursively using Dempster's rule, and the final decision is made through pignistic probability transformation.

## Repository Structure

```text
ClusterBBA/
├── baseline/               # DS-SVM and evidential deep learning baselines
├── cluster/                # Cluster objects, online clustering, scale weights
├── data/                   # BBA datasets, generated data, and BBA generation scripts
├── divergence/             # BJS, Jousselme, RB, RD_CCJS / D_CC-related measures
├── entropy/                # Deng entropy and information-volume utilities
├── experiments/            # Classification, hyperparameter tuning, ablation, statistics
├── experiments_result/     # Saved experimental outputs and tuned parameters
├── figures/                # Figure-generation scripts
├── fractal/                # Maximum-Deng-entropy fractal operator
├── fusion/                 # Dempster, Murphy, Deng, Xiao, and proposed fusion rules
├── mean/                   # Mean BBA and average divergence utilities
├── utility/                # BBA data structure, I/O, probability transform, plotting
├── config.py               # Global hyperparameters and numerical constants
├── main.py                 # Placeholder script; use experiment scripts as entry points
└── requirements.txt        # Python dependencies
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Ma-27/ClusterBBA.git
cd ClusterBBA

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.14 or newer is recommended because the code uses modern Python type-hint syntax. The proposed D-S evidence fusion pipeline does not require a GPU. PyTorch is included mainly for the machine-learning baseline implementations.

## Quick Start

#### 1. Run a lightweight test

```bash
python experiments/application_iris.py --method Proposed --debug
```

The `--debug` flag evaluates only a small number of samples and is useful for checking whether the environment is configured correctly.

#### 2. Run the proposed method on benchmark datasets

```bash
python experiments/application_iris.py  --method Proposed --kfold
python experiments/application_wine.py  --method Proposed --kfold
python experiments/application_seeds.py --method Proposed --kfold
python experiments/application_glass.py --method Proposed --kfold
```

The `--kfold` option uses fold-specific hyperparameters saved under `experiments_result/`, such as `bayes_best_params_kfold_iris.csv`. If the corresponding file is missing, run Bayesian optimization first.

#### 3. Compare with classical D-S evidence-theory baselines

```bash
python experiments/application_iris.py --method Dempster
python experiments/application_iris.py --method Murphy
python experiments/application_iris.py --method Deng
python experiments/application_iris.py --method "Xiao BJS"
python experiments/application_iris.py --method "Xiao RB"
```

Available methods in the application scripts include:

```text
Dempster, Murphy, Deng, Xiao BJS, Xiao BJS Pure, Xiao RB, Proposed
```

#### 4. Tune hyperparameters with Bayesian optimization

```bash
python experiments/tune_hyperparams_bayes.py --dataset iris --kfold --trials 50
```

To jointly optimize $\alpha$ together with $(\lambda, \mu)$:

```bash
python experiments/tune_hyperparams_bayes.py --dataset iris --kfold --trials 50 --alpha
```

Supported dataset names include:

```text
iris, wine, seeds, glass
```

## Minimal Python API Example

```python
from utility.bba import BBA
from fusion.my_rule import my_combine
from utility.probability import pignistic, argmax

frame = {"A", "B", "C"}

m1 = BBA({
  frozenset({"A"}): 0.50,
  frozenset({"B"}): 0.20,
  frozenset({"C"}): 0.30,
}, frame=frame, name="m1")

m2 = BBA({
  frozenset({"A"}): 0.90,
  frozenset({"B"}): 0.10,
}, frame=frame, name="m2")

m3 = BBA({
  frozenset({"A"}): 0.55,
  frozenset({"B"}): 0.10,
  frozenset({"A", "C"}): 0.35,
}, frame=frame, name="m3")

fused = my_combine([m1, m2, m3], lambda_val=1.0, mu_val=1.0)
prob = pignistic(fused)
decision, confidence = argmax(prob)

print("Fused BBA:", fused.to_formatted_dict())
print("Decision:", decision, "Confidence:", confidence)
```

To inspect the intermediate cluster structure:

```python
from cluster.multi_clusters import construct_clusters_by_sequence

mc = construct_clusters_by_sequence([m1, m2, m3], debug=True)
mc.print_all_info()
```

## Experimental Results Reported in the Paper

The paper evaluates the proposed framework on four UCI benchmark datasets using nested five-fold cross-validation. The proposed method is compared with classical D-S evidence-theory methods and machine-learning-based baselines.

| Dataset | Accuracy | Macro F1 | Notes                                                                                                     |
|---------|---------:|---------:|-----------------------------------------------------------------------------------------------------------|
| Iris    |   0.9667 |   0.9667 | Clear class separability; the framework benefits from minority-cluster information.                       |
| Wine    |   0.9663 |   0.9669 | Higher feature dimensionality; hyperparameter tuning is important.                                        |
| Seeds   |   0.9190 |   0.9195 | Moderately overlapping classes; cluster-level weighting remains effective.                                |
| Glass   |   0.5280 |   0.5040 | Strong class imbalance and feature conflict; the proposed method is robust among classical D-S baselines. |

For the Glass dataset, the paper additionally reports that the proposed method obtains precision 0.5328, recall 0.5603, MCC 0.3735, and AUC 0.8266 in the reported setting.

## Main Implementation Map

| Paper concept                                   | Implementation                          |
|-------------------------------------------------|-----------------------------------------|
| Basic Belief Assignment                         | `utility/bba.py`                        |
| Pignistic probability transformation            | `utility/probability.py`                |
| Maximum-Deng-entropy fractal operator           | `fractal/fractal_max_entropy.py`        |
| Single cluster and fractal centroid             | `cluster/one_cluster.py`                |
| Sequential reward-based clustering              | `cluster/multi_clusters.py`             |
| Scale weights $w_p(A)$                          | `cluster/cluster_weights_calculator.py` |
| Cluster-cluster divergence $D_{CC}$ / `RD_CCJS` | `divergence/rd_ccjs.py`                 |
| Proposed cluster-based fusion rule              | `fusion/my_rule.py`                     |
| Classical D-S fusion baselines                  | `fusion/`                               |
| Dataset-level classification evaluation         | `experiments/application_*.py`          |
| Bayesian optimization of $(\lambda, \mu)$       | `experiments/tune_hyperparams_bayes.py` |

## Reproducibility Notes

- The proposed method uses a greedy sequential evidence-assignment rule. Different insertion orders of BBAs may lead to different cluster structures.

- Hyperparameters $(\mu, \lambda)$ are dataset-sensitive. The paper uses Bayesian optimization to select fold-specific values.

- The expert-bias coefficient $\alpha$ controls the trust placed in large clusters. Smaller values can preserve minority-cluster information, while larger values emphasize majority consensus.

- Several scripts save outputs to `experiments_result/`. Existing files in that directory may be reused by the application scripts.

- `main.py` is only a placeholder. For reproduction, use the scripts in `experiments/`.

## Data

The experiments use BBA-form data derived from standard UCI benchmark datasets, including Iris, Wine, Seeds, and Glass. The BBA generation scripts are provided under `data/bba_generation/`, and the application scripts load the generated BBA CSV files from `data/`.

## Citation

If this repository helps your research, please cite the paper:

```bibtex
@article{ma2025clusterbba,
  title = {A Cluster-Level Information Fusion Framework for D-S Evidence Theory with Its Applications in Pattern Classification},
  author = {Ma, Minghao and Fei, Liguo},
  journal = {Mathematics},
  volume = {13},
  number = {19},
  pages = {3144},
  year = {2025},
  publisher = {MDPI}
}
```

## License

The paper is published as an open-access article under the Creative Commons Attribution 4.0 International (CC BY 4.0)
license. The source code in this GitHub repository is released under the MIT License; see the repository-level `LICENSE`
file for details.

## Acknowledgement

This repository accompanies the research article above and is intended to support reproducibility, academic dissemination, and further research on uncertainty reasoning, D-S evidence theory, and interpretable information fusion.

