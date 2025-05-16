# BPA Results

```
"E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\.venv\Scripts\python.exe" "E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\data\data_preparation_test.py"
示例前五行数据:
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa
数据维度: (150, 4)

训练集（共 120 个样本）:
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                5.4               3.9                1.7               0.4  setosa
4                4.6               3.4                1.4               0.3  setosa
...
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     target
115                6.7               3.0                5.2               2.3  virginica
116                6.3               2.5                5.0               1.9  virginica
117                6.5               3.0                5.2               2.0  virginica
118                6.2               3.4                5.4               2.3  virginica
119                5.9               3.0                5.1               1.8  virginica

测试集（共 30 个样本）:
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                4.6               3.1                1.5               0.2  setosa
1                5.0               3.6                1.4               0.2  setosa
2                5.8               4.0                1.2               0.2  setosa
3                5.1               3.8                1.5               0.3  setosa
4                5.1               3.3                1.7               0.5  setosa
...
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     target
25                6.2               2.8                4.8               1.8  virginica
26                7.2               3.0                5.8               1.6  virginica
27                6.9               3.1                5.4               2.1  virginica
28                6.9               3.1                5.1               2.3  virginica
29                6.8               3.2                5.9               2.3  virginica

=== 单类区间数 ===

setosa ->
min  max
sepal length (cm)  4.3  5.7
sepal width (cm)   2.3  4.4
petal length (cm)  1.0  1.9
petal width (cm)   0.1  0.6

versicolor ->
min  max
sepal length (cm)  5.0  6.9
sepal width (cm)   2.2  3.3
petal length (cm)  3.3  5.1
petal width (cm)   1.0  1.8

virginica ->
min  max
sepal length (cm)  4.9  7.9
sepal width (cm)   2.2  3.8
petal length (cm)  4.5  6.9
petal width (cm)   1.4  2.5

=== 组合类交叉区间 ===

[np.str_('setosa'), np.str_('versicolor')] ->
min  max
sepal length (cm)  5.0  5.7
sepal width (cm)   2.3  3.3
petal length (cm)  3.3  1.9
petal width (cm)   1.0  0.6

[np.str_('setosa'), np.str_('virginica')] ->
min  max
sepal length (cm)  4.9  5.7
sepal width (cm)   2.3  3.8
petal length (cm)  4.5  1.9
petal width (cm)   1.4  0.6

[np.str_('versicolor'), np.str_('virginica')] ->
min  max
sepal length (cm)  5.0  6.9
sepal width (cm)   2.2  3.3
petal length (cm)  4.5  5.1
petal width (cm)   1.4  1.8

[np.str_('setosa'), np.str_('versicolor'), np.str_('virginica')] ->
min  max
sepal length (cm)  5.0  5.7
sepal width (cm)   2.3  3.3
petal length (cm)  4.5  1.9
petal width (cm)   1.4  0.6

=== 单样本各属性的BPA（退化区间） ===

属性 sepal length (cm):
setosa          : 0.2141
versicolor      : 0.0993
virginica       : 0.0749
setosa,versicolor : 0.1685
setosa,virginica : 0.1756
versicolor,virginica : 0.0993
setosa,versicolor,virginica : 0.1685
概率和 Σ = 1.0000

属性 sepal width (cm):
setosa          : 0.1119
versicolor      : 0.1423
virginica       : 0.1423
setosa,versicolor : 0.1553
setosa,virginica : 0.1505
versicolor,virginica : 0.1423
setosa,versicolor,virginica : 0.1553
概率和 Σ = 1.0000

属性 petal length (cm):
setosa          : 0.7172
versicolor      : 0.1130
virginica       : 0.0748
setosa,versicolor : 0.0000
setosa,virginica : 0.0000
versicolor,virginica : 0.0951
setosa,versicolor,virginica : 0.0000
概率和 Σ = 1.0000

属性 petal width (cm):
setosa          : 0.5722
versicolor      : 0.1642
virginica       : 0.1180
setosa,versicolor : 0.0000
setosa,virginica : 0.0000
versicolor,virginica : 0.1455
setosa,versicolor,virginica : 0.0000
概率和 Σ = 1.0000

随机划分一次的准确率: 0.900

进程已结束，退出代码为 0
```

