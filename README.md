# 🗃 AI-Paper-Drawer
人工智能论文关键点概括集结。This project aims to collect key points of AI papers. 扫码加入QQ交流群832405795 ↓


![](drawer/home.png)

此 repo 旨在记录各 AI 论文的精简概括，以便大家更好地了解不同邻域的发展脉络。

# 子抽屉
[图神经网络](图网络专区.md)

# 💫 Graph 图推理
## 点云语义分割
### [【2017 CVPR】](https://arxiv.org/abs/1612.00593) PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- `动机：`构造具有排列不变性的神经网络
- 本文开创 DL 在无序点云上识别的先河，利用核长为1的卷积核对每个点单独升维后使用对称函数（+、max 等）获取具有输入排列不变性的全局点云特征
### [【2018 CVPR】](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf) Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling
- **动机：**推广卷积到三维离散点云
- 类比卷积核对分布相近数据具有更高激活值的特点，构造可学习的图核，通过局部区域点的分布与图核的相似性计算激活值
### [【2019 CVPR】](https://engineering.purdue.edu/~jshan/publications/2018/Lei%20Wang%20Graph%20Attention%20Convolution%20for%20Point%20Cloud%20Segmentation%20CVPR2019.pdf) Graph Attention Convolution for Point Cloud Segmentation
- **动机：**引入注意力机制缓解图卷积各向同性问题，避免特征污染
- 将离散卷积核设定为相对位置和特征差分的函数，并利用 `softmax` 做归一化

# 🖼 CV 计算机视觉
## 卷积演变
### [【2019 CVPR】](https://arxiv.org/abs/1904.05049v3) Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
- **动机：**

# 📜 NLP 自然语言处理

# Recommendation 推荐系统

# 👾 RL 强化学习

# 🎨 GANs 生成式对抗

# ⚗ Others 其他

# 其他
- 想学 🐍 Python？欢迎光临[ LeetCode最短Python题解 ](https://github.com/cy69855522/Shortest-LeetCode-Python-Solutions)，和我们深入探索 Python 特性。
