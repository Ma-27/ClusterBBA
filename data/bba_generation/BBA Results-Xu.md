# BBA Results-Xu

## IRIS 数据集结果

```
"E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\.venv\Scripts\python.exe" "E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\data\bba_generation\xu_iris.py" 
数据集包含 3 类，4 个属性。

样本 0 - 特征: [5.1, 3.5, 1.4, 0.2]，标签: Setosa
样本 1 - 特征: [4.9, 3.0, 1.4, 0.2]，标签: Setosa
样本 2 - 特征: [4.7, 3.2, 1.3, 0.2]，标签: Setosa
样本 3 - 特征: [4.6, 3.1, 1.5, 0.2]，标签: Setosa
样本 4 - 特征: [5.0, 3.6, 1.4, 0.2]，标签: Setosa
总样本数（全体 X）= 150
类别标签（full_class_names）: ['Setosa', 'Versicolor', 'Virginica']
属性名称（attr_names）: ['SL', 'SW', 'PL', 'PW']

Calculating means and stds: 100%|██████████| 3/3 [00:00<?, ?it/s]
Generating BBA:   0%|          | 0/150 [00:00<?, ?it/s]
Training Set Normality Indices (1 表示拒绝正态假设):
            SL  SW  PL  PW
Setosa       0   0   0   1
Versicolor   0   0   0   0
Virginica    0   0   0   0

需 Box-Cox 变换的属性: []
Generating BBA: 100%|██████████| 150/150 [00:00<00:00, 2837.67it/s]

已保存格式化后的 BBA 至 E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\data\iris_bba.csv  (共 600 行)

—— 随机抽取 8 个样本对应的 BBA 预览 (每个样本 4 行，共 32 行) ——
 sample_index ground_truth dataset_split attribute  attribute_data  {Vi}  {Ve}   {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           73       Setosa         train        SL             6.1   0.0   0.0 0.9513        0.0      0.000     0.0429          0.0058
           73       Setosa         train        SW             2.8   0.0   0.0 0.5800        0.0      0.321     0.0000          0.0990
           73       Setosa         train        PL             4.7   0.0   0.0 1.0000        0.0      0.000     0.0000          0.0000
           73       Setosa         train        PW             1.2   0.0   0.0 1.0000        0.0      0.000     0.0000          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data  {Vi}   {Ve}   {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           18   Versicolor         train        SL             5.7   0.0 0.5238 0.0000     0.4739     0.0000        0.0          0.0023
           18   Versicolor         train        SW             3.8   0.0 0.0000 0.4737     0.0000     0.3792        0.0          0.1471
           18   Versicolor         train        PL             1.7   0.0 0.7055 0.0000     0.2945     0.0000        0.0          0.0000
           18   Versicolor         train        PW             0.3   0.0 0.6931 0.0000     0.3069     0.0000        0.0          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data   {Vi}   {Ve}  {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
          118    Virginica         train        SL             7.7 0.5381 0.0000   0.0     0.4609        0.0        0.0          0.0009
          118    Virginica         train        SW             2.6 0.0000 0.4738   0.0     0.4044        0.0        0.0          0.1218
          118    Virginica         train        PL             6.9 0.9707 0.0000   0.0     0.0293        0.0        0.0          0.0000
          118    Virginica         train        PW             2.3 0.9989 0.0000   0.0     0.0011        0.0        0.0          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data  {Vi}   {Ve}  {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           78   Versicolor         train        SL             6.0   0.0 0.5073   0.0     0.0000        0.0     0.3871          0.1056
           78   Versicolor         train        SW             2.9   0.0 0.6991   0.0     0.2593        0.0     0.0000          0.0416
           78   Versicolor         train        PL             4.5   0.0 0.9874   0.0     0.0126        0.0     0.0000          0.0000
           78   Versicolor         train        PW             1.5   0.0 0.9976   0.0     0.0024        0.0     0.0000          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data   {Vi}  {Ve}   {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           76    Virginica         train        SL             6.8 0.7340   0.0 0.0000     0.2660     0.0000        0.0          0.0000
           76    Virginica         train        SW             2.8 0.0000   0.0 0.4737     0.0000     0.3792        0.0          0.1471
           76    Virginica         train        PL             4.8 0.9827   0.0 0.0000     0.0173     0.0000        0.0          0.0000
           76    Virginica         train        PW             1.4 1.0000   0.0 0.0000     0.0000     0.0000        0.0          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data   {Vi}   {Ve}  {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           31   Versicolor         train        SL             5.4 0.0000 0.6293   0.0     0.3585        0.0        0.0          0.0122
           31   Versicolor         train        SW             3.4 0.4448 0.0000   0.0     0.3378        0.0        0.0          0.2175
           31   Versicolor         train        PL             1.5 0.0000 0.7866   0.0     0.2134        0.0        0.0          0.0000
           31   Versicolor         train        PW             0.4 0.0000 0.9604   0.0     0.0396        0.0        0.0          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data   {Vi}  {Ve}  {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
           64    Virginica         train        SL             5.6 0.9885   0.0   0.0     0.0115        0.0        0.0          0.0000
           64    Virginica         train        SW             2.9 0.4448   0.0   0.0     0.3378        0.0        0.0          0.2175
           64    Virginica         train        PL             3.6 0.9999   0.0   0.0     0.0001        0.0        0.0          0.0000
           64    Virginica         train        PW             1.3 0.9989   0.0   0.0     0.0011        0.0        0.0          0.0000
 sample_index ground_truth dataset_split attribute  attribute_data   {Vi}   {Ve}  {Se}  {Vi ∪ Ve}  {Vi ∪ Se}  {Ve ∪ Se}  {Vi ∪ Ve ∪ Se}
          141    Virginica          test        SL             6.9 0.0000 0.5936   0.0     0.0000        0.0      0.266          0.1404
          141    Virginica          test        SW             3.1 0.0000 0.4738   0.0     0.4044        0.0      0.000          0.1218
          141    Virginica          test        PL             5.1 0.5055 0.0000   0.0     0.4945        0.0      0.000          0.0000
          141    Virginica          test        PW             2.3 0.9939 0.0000   0.0     0.0061        0.0      0.000          0.0000

进程已结束，退出代码为 0
```