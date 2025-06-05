# BBA Results-Kang

## IRIS 数据集结果

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

## DRY BEANS 数据集，全量合并

```
"E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\.venv\Scripts\python.exe" "E:\CScodes\Research\Information Gain Driven Fusion of Conflicting Evidence\data\data_preparation_drybeans.py"
检测到已有文件：Dry_Beans_Dataset.csv，将直接加载...
前5行数据如下：
Area  Perimeter  MajorAxisLength  ...  ShapeFactor3  ShapeFactor4  Class
0  28395    610.291       208.178117  ...      0.834222      0.998724  SEKER
1  28734    638.018       200.524796  ...      0.909851      0.998430  SEKER
2  29380    624.110       212.826130  ...      0.825871      0.999066  SEKER
3  30008    645.884       210.557999  ...      0.861794      0.994199  SEKER
4  30140    620.134       201.847882  ...      0.941900      0.999166  SEKER

[5 rows x 17 columns]
示例前五行数据:
Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  Compactness  ShapeFactor1  ShapeFactor2  ShapeFactor3  \
0  28395.0    610.291       208.178117       173.888747      1.197191      0.549812     28715.0     190.141097  0.763923  0.988856   0.958027     0.913358      0.007332      0.003147      0.834222   
1  28734.0    638.018       200.524796       182.734419      1.097356      0.411785     29172.0     191.272750  0.783968  0.984986   0.887034     0.953861      0.006979      0.003564      0.909851   
2  29380.0    624.110       212.826130       175.931143      1.209713      0.562727     29690.0     193.410904  0.778113  0.989559   0.947849     0.908774      0.007244      0.003048      0.825871   
3  30008.0    645.884       210.557999       182.516516      1.153638      0.498616     30724.0     195.467062  0.782681  0.976696   0.903936     0.928329      0.007017      0.003215      0.861794   
4  30140.0    620.134       201.847882       190.279279      1.060798      0.333680     30417.0     195.896503  0.773098  0.990893   0.984877     0.970516      0.006697      0.003665      0.941900

ShapeFactor4 target  
0      0.998724  SEKER  
1      0.998430  SEKER  
2      0.999066  SEKER  
3      0.994199  SEKER  
4      0.999166  SEKER  
数据维度: (13611, 16)

训练集（共 5441 个样本）:
Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  Compactness  ShapeFactor1  ShapeFactor2  ShapeFactor3  \
0  30008.0    645.884       210.557999       182.516516      1.153638      0.498616     30724.0     195.467062  0.782681  0.976696   0.903936     0.928329      0.007017      0.003215      0.861794   
1  30140.0    620.134       201.847882       190.279279      1.060798      0.333680     30417.0     195.896503  0.773098  0.990893   0.984877     0.970516      0.006697      0.003665      0.941900   
2  30477.0    670.033       211.050155       184.039050      1.146768      0.489478     30970.0     196.988633  0.762402  0.984081   0.853080     0.933374      0.006925      0.003242      0.871186   
3  30519.0    629.727       212.996755       182.737204      1.165591      0.513760     30847.0     197.124320  0.770682  0.989367   0.967109     0.925480      0.006979      0.003158      0.856514   
4  31158.0    641.105       212.066975       187.192960      1.132879      0.469924     31474.0     199.177302  0.781313  0.989960   0.952623     0.939219      0.006806      0.003267      0.882132

ShapeFactor4 target  
0      0.994199  SEKER  
1      0.999166  SEKER  
2      0.999049  SEKER  
3      0.998345  SEKER  
4      0.999349  SEKER  
...
Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  Compactness  ShapeFactor1  ShapeFactor2  \
5436  41995.0    765.763       284.073178       188.591957      1.506285      0.747835     42477.0     231.235150  0.732514  0.988653   0.899951     0.813999      0.006764      0.001832   
5437  42008.0    759.454       280.332717       191.218136      1.466036      0.731248     42419.0     231.270938  0.711710  0.990311   0.915248     0.824987      0.006673      0.001907   
5438  42049.0    770.185       290.163403       185.051685      1.568013      0.770243     42503.0     231.383771  0.756005  0.989318   0.890790     0.797426      0.006901      0.001721   
5439  42070.0    760.701       276.691651       193.945366      1.426647      0.713216     42458.0     231.441543  0.730813  0.990862   0.913596     0.836460      0.006577      0.001986   
5440  42139.0    759.321       281.539928       191.187979      1.472582      0.734065     42569.0     231.631261  0.729932  0.989899   0.918424     0.822730      0.006681      0.001888

      ShapeFactor3  ShapeFactor4    target  
5436      0.662594      0.998055  DERMASON  
5437      0.680604      0.997790  DERMASON  
5438      0.635888      0.997080  DERMASON  
5439      0.699666      0.998176  DERMASON  
5440      0.676884      0.996767  DERMASON

测试集（共 8170 个样本）:
Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  Compactness  ShapeFactor1  ShapeFactor2  ShapeFactor3  \
0  28395.0    610.291       208.178117       173.888747      1.197191      0.549812     28715.0     190.141097  0.763923  0.988856   0.958027     0.913358      0.007332      0.003147      0.834222   
1  28734.0    638.018       200.524796       182.734419      1.097356      0.411785     29172.0     191.272750  0.783968  0.984986   0.887034     0.953861      0.006979      0.003564      0.909851   
2  29380.0    624.110       212.826130       175.931143      1.209713      0.562727     29690.0     193.410904  0.778113  0.989559   0.947849     0.908774      0.007244      0.003048      0.825871   
3  30279.0    634.927       212.560556       181.510182      1.171067      0.520401     30600.0     196.347702  0.775688  0.989510   0.943852     0.923726      0.007020      0.003153      0.853270   
4  30685.0    635.681       213.534145       183.157146      1.165852      0.514081     31044.0     197.659696  0.771561  0.988436   0.954240     0.925658      0.006959      0.003152      0.856844

ShapeFactor4 target  
0      0.998724  SEKER  
1      0.998430  SEKER  
2      0.999066  SEKER  
3      0.999236  SEKER  
4      0.998953  SEKER  
...
Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  Compactness  ShapeFactor1  ShapeFactor2  \
8165  42070.0    763.489       289.022373       186.123434      1.552853      0.765046     42556.0     231.441543  0.768823  0.988580   0.906936     0.800774      0.006870      0.001743   
8166  42097.0    759.696       288.721612       185.944705      1.552728      0.765002     42508.0     231.515799  0.714574  0.990331   0.916603     0.801865      0.006858      0.001749   
8167  42101.0    757.499       281.576392       190.713136      1.476439      0.735702     42494.0     231.526798  0.799943  0.990752   0.922015     0.822252      0.006688      0.001886   
8168  42147.0    763.779       283.382636       190.275731      1.489326      0.741055     42667.0     231.653248  0.705389  0.987813   0.907906     0.817457      0.006724      0.001852   
8169  42159.0    772.237       295.142741       182.204716      1.619841      0.786693     42600.0     231.686223  0.788962  0.989648   0.888380     0.784997      0.007001      0.001640

      ShapeFactor3  ShapeFactor4    target  
8165      0.641239      0.995750  DERMASON  
8166      0.642988      0.998385  DERMASON  
8167      0.676099      0.998219  DERMASON  
8168      0.668237      0.995222  DERMASON  
8169      0.616221      0.998180  DERMASON  
Dempster 组合: 100%|████████████████████████| 8170/8170 [37:23<00:00,  3.64it/s]
Dry Beans 数据集分类准确率: 0.2444

进程已结束，退出代码为 0
```
