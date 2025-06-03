## 训练-测试划分

这几行代码分为两步：

1. **划分训练集和测试集**

   ```
   X_tr, X_te, y_tr, y_te = train_test_split(
       X_all, y_all, train_size=TRAIN_RATIO, stratify=y_all, random_state=42
   )
   ```

    - `X_all`：原始的 Iris 样本特征矩阵，形状是 `(150, 4)`，其中 150 行对应 150 个鸢尾花样本，4 列对应四个属性（SL、SW、PL、PW）。
    - `y_all`：原始的 Iris 类别标签数组，长度为 150，每个元素是 0/1/2 中的一个整数，分别代表 Setosa、Versicolor、Virginica。
    - `train_test_split(...)`：这是 scikit-learn 提供的常用函数，用来把数据随机分成两部分——“训练集”(train) 和 “测试集”(
      test)。
        - `train_size=TRAIN_RATIO` 表示我们希望保留多少比例的样本用作“训练”——在这里 `TRAIN_RATIO=0.6`，所以训练集占
          60%，测试集占 40%。
        - `stratify=y_all` 表示在随机打乱（shuffle）并切分的时候，会保证训练集和测试集里各类别的样本比例与原始数据集一致（也就是保证三类在训练/测试里的分布比例相同）。
        - `random_state=42` 是随机种子，用于让每次运行这个 split 都能得到同样的划分结果，便于实验可复现。
    - 最终输出：
        - `X_tr`：训练集的特征矩阵，形状约为 `(90, 4)`（150×0.6 约等于 90）。
        - `X_te`：测试集的特征矩阵，形状约为 `(60, 4)`（150×0.4 约等于 60）。
        - `y_tr`：训练集对应的标签数组，长度为 90。
        - `y_te`：测试集对应的标签数组，长度为 60。

2. **将 NumPy 数组封装成自定义 Dataset**

   ```
   train_dataset = IrisDataset(X_tr, y_tr, attr_names, class_names)
   test_dataset  = IrisDataset(X_te, y_te, attr_names, class_names)
   ```

   这里我们使用了一个自定义的 PyTorch 风格的 `Dataset` 类（`IrisDataset`）来将这些 NumPy 数据包装起来。具体做了什么：

    - `IrisDataset.__init__(self, data, targets, attr_names, class_names)`
        - `data`：一个二维的 NumPy 数组（在训练集里就是 `X_tr`，测试集里就是 `X_te`）。
        - `targets`：一个一维的 NumPy 数组（训练集对应 `y_tr`，测试集对应 `y_te`）。
        - `attr_names`：一个长度为 4 的字符串列表，形如 `["SL", "SW", "PL", "PW"]`，表示每一列（每个属性）的名称。
        - `class_names`：一个长度为 3 的字符串列表，形如 `["Se", "Ve", "Vi"]`，表示类别对应的缩写标签。
        - 在 `__init__` 里，这些输入会被分别存到实例变量 `self.data`、`self.targets`、`self.attr_names`、`self.class_names`
          中。此外，它还会记录 `self.n_samples = data.shape[0]`，也就是有多少条样本；`self.n_attr = data.shape[1]`
          ，也就是每条样本有几个属性（这里永远是 4）。
    - `IrisDataset` 里最主要的两个方法：
        1. `__len__(self)` 返回样本总数，对训练集就是 `len(train_dataset) == 90`，对测试集就是 `len(test_dataset) == 60`。
        2. `__getitem__(self, idx)`：给定一个索引 `idx`，它会返回 `(self.data[idx], self.targets[idx])`，即一个形如
           `(array_of_shape_(4,), int)` 的元组——前者是第 `idx` 条样本的四个属性数值，后者是该样本对应的类别索引（0/1/2）。
    - 除此之外，`IrisDataset` 还提供了一个 `get_sample(self, idx)` 方法，功能与 `__getitem__` 类似，也是返回对应索引的
      `(x_vec, y_val)`，区别只是我们在 BBA 生成步骤里，用 `get_sample` 显示地拿到这一行的特征向量和标签，而不是让 PyTorch
      DataLoader 或其他框架去自动迭代。

   因此，当我们写了这两行后：

   ```
   train_dataset = IrisDataset(X_tr, y_tr, attr_names, class_names)
   test_dataset  = IrisDataset(X_te, y_te, attr_names, class_names)
   ```

   就相当于完成了这几个动作：

    1. 把训练集（90×4 的特征矩阵 + 90 的标签）和测试集（60×4 + 60）分别打包成两个能够通过索引访问的对象。
    2. `train_dataset.data` 会是一个形状 `(90, 4)` 的 `float32` 数组，`train_dataset.targets` 是形状 `(90,)` 的 `int64`
       数组；同理 `test_dataset.data` 是 `(60, 4)`，`test_dataset.targets` 是 `(60,)`。
    3. 你可以对 `train_dataset` 调用 `len(train_dataset)` 或者 `train_dataset[i]`，其中 `train_dataset[i]` 会拿到第 `i`
       个训练样本及其标签；同理对 `test_dataset` 也可以这么操作。

#### 小结

- 这两行代码的第一部分负责“**随机且分层**（stratified）地把原始 150 条样本切成 60% 的训练 + 40% 的测试”。
- 第二部分是“把 NumPy 原始数组封装成一个自定义的 `Dataset` 对象”，使得后续在生成 BBA、或者如果日后想用 PyTorch 的
  `DataLoader`，都可以通过一句 `dataset[i]` 拿到第 `i` 条样本及其标签，而不用手动管理索引与数据结构。

封装完之后，`train_dataset`、`test_dataset` 的内部结构就是：

```
train_dataset.data       # 形状 (90, 4)，dtype=float32
train_dataset.targets    # 形状 (90,), dtype=int64
train_dataset.attr_names # ["SL", "SW", "PL", "PW"]
train_dataset.class_names# ["Se", "Ve", "Vi"]
train_dataset.n_samples  # 90
train_dataset.n_attr     # 4

test_dataset.data        # 形状 (60, 4)
test_dataset.targets     # 形状 (60,)
... 其余属性同理 ...
```

这样在后续写 BBA 生成逻辑时，就可以直接用 `train_dataset.get_sample(idx)` 或者 `for x_vec, y_val in train_dataset:`
的方式来依次遍历训练集中每一个样本，而不用再关注它们在大数组 `X_tr`、`y_tr` 中的具体索引。



