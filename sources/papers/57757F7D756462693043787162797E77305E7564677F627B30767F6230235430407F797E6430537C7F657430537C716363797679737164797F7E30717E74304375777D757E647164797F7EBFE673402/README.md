# [Geometry Sharing Network for 3D Point Cloud Classification and Segmentation](https://arxiv.org/abs/1912.10644v1)

## 背景
- 构建几何结构相似点间的连接
- 本文卷积核与DGCNN差不多，只是在边特征上多拼了一部分特征空间相近点的特征
## 模型流程
### 特征空间的KNN
- 首先在欧式空间中找到K个邻居，以中心点为原点，得到局部特征矩阵`M`，`M ∈ [K, C] = x_j - x_i`(C代表通道数)
- 作者定义局部结构的特征矩阵为`(M^T)M ∈ [C, C]`
- 通过特征分解得到有序的特征值`eigenvalue ∈ [1, C]`
- 每个点都有这样的一个局部结构的特征值，在这个空间进行KNN搜索即可
### 卷积核
- DGCNN：`Max(MLP(Cat(x_i, x_j - x_i)))`，`x_i`代表中心点的特征，`x_j`代表邻点特征
- GSNet：`Max(MLP(Cat(x_j, x_j - x_i, x_j_eig, x_j_eig - x_i)))`，`x_j_eig`代表特征值空间中的邻点特征
- GSNet中欧式空间的邻点特征`Cat(x_j, x_j - x_i) ∈ [K, C]`（K代表邻居个数，C代表通道数），和特征值空间的邻点特征`Cat(x_j_eig, x_j_eig - x_i) ∈ [K, C]`是直接拼接的。也就是说把欧式空间第一近的邻居特征和特征空间第一近的特征拼接在一起作为第一个邻居的特征，以此类推。所以GSNet虽然在欧式和特征空间都搜索了K个邻居（一共有2K个），实际计算的时候只有K个邻居。
### Drawbacks
- 局部结构特征分解慢
