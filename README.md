# 🗃 AI-Paper-Drawer
人工智能论文笔记，若有不当之处欢迎指正(发 issue 或 PR)。 ⛄欢迎扫码加入QQ交流群832405795 ↓


![](drawer/home.png)

此 repo 旨在记录各 AI 论文具有启发性的核心思想和流程
- 点击论文标题前的超链接可访问原文
- 点击✒可进入流程速记页面，记录核心算法公式，便于复习

# 子抽屉
[图神经网络](图网络专区.md)

## 相关链接
- 🐍 想学Python？欢迎光临[ LeetCode最短Python题解 ](https://github.com/cy69855522/Shortest-LeetCode-Python-Solutions)，和我们深入探索 Python 特性。
- [🚀 AI Power](https://www.aipower.xyz) 云GPU租借/出租平台：图网络的计算需要高算力支持，赶论文、拼比赛的朋友不妨了解一下~ 现在注册并绑定（参考Github）即可获得高额算力，注册不涉及个人隐私信息，奖励可随时提现。详情请参考[AI Power指南](https://github.com/cy69855522/AI-Power)

# 💫 Graph 图网络
## 图数据
### [【2016 ICLR】](https://arxiv.org/pdf/1511.05493.pdf) [✒](sources/papers/57514455543057425140583043554145555E5355305E554542515C305E5544475F425B43BFE673402/README.md) GATED GRAPH SEQUENCE NEURAL NETWORKS
- `动机：为了使GNN能够用于处理序列问题`
- 图神经网络的一种，以每一次局部传播的结果作为输入，网络层数即传播次数固定，层与层之间的信息传递手法利用GRU的门控机制
## 点云
### [【2020 AAAI】](https://arxiv.org/abs/1912.10775) [✒](sources/papers/407F797E64225E7F74752A30537F6262757C7164797F7E305C7571627E797E77307F763054697E717D79733D5E7F747530767F6230407F797E6430537C7F65743056757164656275305D7F74757C797E771DBFE673402/README.md) Point2Node: Correlation Learning of Dynamic-Node for Point Cloud Feature Modeling
- `动机：探索自我(自身特征通道)相关性、局部相关性、非局部相关性`
- 利用`softmax`引入自身通道注意力、节点与节点间注意力。考虑节点与节点间注意力时参考“Non-Local Neural Network”做矩阵乘法构建各点间的注意力。利用门控式分权聚合代替残差连接
### [【2020 AAAI】](https://arxiv.org/abs/1912.10644v1) [✒](sources/papers/57757F7D756462693043787162797E77305E7564677F627B30767F6230235430407F797E6430537C7F657430537C716363797679737164797F7E30717E74304375777D757E647164797F7EBFE673402/README.md) Geometry Sharing Network for 3D Point Cloud Classification and Segmentation
- `动机：构建特征空间的相似连接，挖掘远距离相似结构的相关性`
- 利用局部点构成的结构矩阵的特征值作为旋转平移不变的局部特征，寻找结构相似的点作为邻居
### [【2019 ICCV】](https://arxiv.org/abs/1908.04512) [](sources/papers/597E647562607F7C7164757430537F7E667F7C6564797F7E717C305E7564677F627B6330767F6230235430407F797E6430537C7F657430457E7475626364717E74797E77BFE673402/README.md) Interpolated Convolutional Networks for 3D Point Cloud Understanding
- `动机：利用插值解决点云数据结构的稀疏性、不规则性和无序性`
- 预设几个离散卷积核权重的位置，对每个中心点所对应的核权重位置进行插值并归一化，然后计算激活值
### [【2019 ICCV】](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_PointCloud_Saliency_Maps_ICCV_2019_paper.pdf) [](sources/papers/407F797E64537C7F65743043717C79757E7369305D716063BFE673402/README.md) PointCloud Saliency Maps
- `动机：建立点云的显著性图，评估每个点对于下游任务的重要性`
- 将某点的坐标移动到原点，计算模型性能差异作为点对于下游任务的贡献度。贡献度由`loss`对于点坐标模长`r`的偏导数决定
### [【2019 CVPR】](https://arxiv.org/abs/1811.07782) [✒](sources/papers/5D7F74757C797E77305C7F73717C3057757F7D756462797330436462657364656275307F7630235430407F797E6430537C7F657463306563797E773057757F3D535E5EBFE673402/README.md) Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN
- `动机：显式建模局部点间的几何结构`
- 将局部点云特征提取过程按三个正交基分解，然后根据边向量与基之间的夹角对提取的特征进行聚合，鼓励网络在整个特征提取层次中保持欧氏空间的几何结构
### [【2019 CVPR】](https://engineering.purdue.edu/~jshan/publications/2018/Lei%20Wang%20Graph%20Attention%20Convolution%20for%20Point%20Cloud%20Segmentation%20CVPR2019.pdf) Graph Attention Convolution for Point Cloud Segmentation
- `动机：引入注意力机制缓解图卷积各向同性问题，避免特征污染`
- 将离散卷积核设定为相对位置和特征差分的函数，并利用 `softmax` 做归一化
### [【2018 CVPR】](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf) Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling
- `动机：类比卷积局部激活性到三维离散点云核相关`
- 类比卷积核对分布相近数据具有更高激活值的特点，构造可学习的图核，通过局部区域点的分布与图核的相似性计算激活值
### [【2018 CVPR】](https://arxiv.org/abs/1711.08920v2) [](sources/papers/43607C797E75535E5E2A30567163643057757F7D75646279733054757560305C7571627E797E77306779647830537F7E64797E657F656330523D43607C797E75305B75627E757C63BFE673402/README.md) SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels
- `动机：一个新的基于b样条的卷积算子，它使得计算时间独立于核大小`
### [【2017 CVPR】](https://arxiv.org/abs/1612.00593) ⭐ PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- `动机：构造具有排列不变性的神经网络`
- 本文开创 DL 在无序点云上识别的先河，利用核长为1的卷积核对每个点单独升维后使用对称函数（+、max 等）获取具有输入排列不变性的全局点云特征
# 🖼 CV 计算机视觉
## 卷积演变
### [【2019 CVPR】](https://arxiv.org/abs/1904.05049v3) Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
- `动机：缓解卷积层在特征图空间频率的冗余`
- 将卷积通道划分为俩个部分，高分辨率通道存储高频特征，低分辨率通道存储低频特征，提高效率

# 📜 NLP 自然语言处理
## 循环神经网络
### [【2014】](https://arxiv.org/abs/1406.1078) [✒](sources/papers/5C7571627E797E773040786271637530427560627563757E647164797F7E63306563797E7730425E5E30557E737F7475623D5475737F74756230767F6230436471647963647973717C305D717378797E75304462717E637C7164797F7EBFE673402/README.md) Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- 提出了`GRU`，其效果与`LSTM`相近，效率更高

# 💞 Recommendation 推荐系统

# 👾 RL 强化学习

# 🎨 GANs 生成式对抗网络

# 🔘 Meta Learning 元学习

# 🚥 Cluster 聚类
## 目标函数
### [【2019 ICCV】](https://arxiv.org/abs/1807.06653v4) [✒](sources/papers/597E66716279717E6430597E767F627D7164797F7E30537C6563647562797E7730767F6230457E6365607562667963757430597D71777530537C716363797679737164797F7E30717E74304375777D757E647164797F7EBFE673402/README.md) Invariant Information Clustering for Unsupervised Image Classification and Segmentation
- `动机：提出一种新的聚类目标IIC作为端到端神经网络损失函数`
- 以一对近似样本投入神经网络获得成对的输出，最大化俩者的互信息

# ⚗ Others 其他

# 🎯 知识点速记
## 评估指标
- accuracy 正确率：被分对的样本 / 所有样本
- precision 精度：分对的正样本 / 预测为正的样本
- recall 召回率（真阳性率）：分对的正样本 / 正样本，有病的被查出来的概率
- 假阳性率：分错的负样本 / 负样本，没病的被当成有病的概率
- ROC曲线：滑动归类阈值来产生关键点并连接，横坐标为`1 - 假阳性率`，纵坐标为`真阳性率`，线下面积`AUC = (1 - 假阳性率)*真阳性率`越高越好
- f1 score：`2*precision*recall / (precision + recall)`
## 优化方法
- 最小二乘法：设偏导为0求解参数
- 梯度下降：朝着损失下降最快的方向迭代
## 损失函数
###  [✒](sources/keyPoints/53627F636330557E64627F6069BFE673402/README.md) Cross Entropy
- 交叉熵常用于分类问题，表示的是预测结果的概率分布与实际结果概率分布的差异

