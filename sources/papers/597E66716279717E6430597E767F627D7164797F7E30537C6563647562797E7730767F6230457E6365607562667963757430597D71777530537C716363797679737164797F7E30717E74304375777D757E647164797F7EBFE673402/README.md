# [Invariant Information Clustering for Unsupervised Image Classification and Segmentation](https://arxiv.org/abs/1807.06653v4)

## 背景
- 
## IIC损失公式
![](f1.png)
- 设`Φ`为神经网络，同时通过两个近似输入`x`和`x'`
![](f2.png)
- 
![](f3.png)

![](code.png)
- 
## 要点记录
- `x'`可以为`x`经过数据增强后的结果，在语义分割中，`x`为某一局部感受野范围的中心像素及分类的最小单位，`x'`也可以是`x`的临近像素
