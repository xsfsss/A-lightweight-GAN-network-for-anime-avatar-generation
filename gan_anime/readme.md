# 这是一个GAN网络的动漫头像生成项目（轻便版）

### A lightweight GAN network for anime avatar generation

Author：xsfsss

## 总览

<small>Overview</small>

这是一个简单的GAN动漫头像生成网络，包含两种类型的头像生成策略：

<small>This is a simple GAN network for generating anime avatars, which includes two types of avatar generation strategies</small>

- 全连接网络   	Fully connected network
- 卷积网络           CNN network

分别对 64x64、128x128、512x512 动漫数据集进行了训练与生成。其中在 64x64 和 128x128 的数据集上，结果十分显著。但由于网络架构问题，在 512x512 的训练过程中出现了一些问题：

<small>The network has been trained and generated on anime datasets of 64x64,  128x128, and 512x512 resolutions. The results are significant for the  64x64 and 128x128 datasets. However, there were some issues during the  512x512 training process due to network architecture problems</small>

- 无法提取到全局特征    Unable to extract global features
- 训练后期模型不收敛    The model did not converge in the later stages of training

本实验与 CSDN 上的 Anime-face_generate 进行了对比。

<small>This experiment was compared to Anime-face_generate on CSDN</small>

​										

![1](readmeimgs\1.png)

## 训练过程数据展示

<small>Training process display</small>

### 512x512 的 CnnGAN

<small>512x512 CnnGAN</small>

![2](readmeimgs\2.png)

### 128x128 的 CnnGAN:

<small>128x128 CnnGAN</small>

![3](readmeimgs\3.png)调整后：

![4](readmeimgs\4.png)

### 128x128 的全连接:

<small>128x128 Fully connected</small>

![5](readmeimgs\5.png)

## To start

### 环境配置

<small>Environment configuration</small>

- python >= 3.6
- torch >= 1.10 + 对应 cuda
- visdom

### 开启 visdom

<small>Start visdom</small>

```
python
python visdom.server -m
```

## 缺点

<small>Disadvantages</small>

网络结构还需要优化，才能应对高清的图片，以及后期的 loss 波动处理尚未解决。

The network structure still needs optimization to deal with  high-resolution images, and the handling of loss fluctuations in the  later stages of training has not been resolved.
