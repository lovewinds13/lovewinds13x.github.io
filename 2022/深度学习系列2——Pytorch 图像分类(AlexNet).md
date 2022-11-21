# 1. 概述

本文主要是参照 B 站 UP 主 [霹雳吧啦Wz](https://space.bilibili.com/18161609) 的视频学习笔记，参考的相关资料在文末**参照**栏给出，包括实现代码和文中用的一些图片。

**整个工程已经上传个人的 github [https://github.com/lovewinds13/QYQXDeepLearning](https://github.com/lovewinds13/QYQXDeepLearning) ，下载即可直接测试，数据集文件因为比较大，已经删除了，按照下文教程下载即可。**

论文下载：[ImageNet Classification with Deep Convolutional Neural Networks
](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

# 2. AlexNet

AlexNet 是 2012 年 ISLVRC 2012（ImageNet Large Scale Visual Recognition Challenge） 竞赛的冠军网络， 分类准确率由传统的 70%+ 提升到 80%+。它是由 Hinton 和他的学生Alex Krizhevsky设计的。 

AlexNet 2012 年在大规模图像识别中一起绝尘，从此引领了深度神经网络的热潮。另外，AlexNet 提出的 ReLU 激活函数， LRN， GPU 加速，数据增强，Dropout 失活部分神经元等方式，深刻的影响了后续的神经网络。

## 2.1 网络框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/65517181ecc44f59a61d75476534a62e.png#pic_center)

AlexNet 共有 8 层组成，其中包括 5 个卷积层，3 个全连接层。各层参数如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2889f07a00864f6c9e042b91adbd5d82.png#pic_center)
计算公式参考[深度学习系列1——Pytorch 图像分类(LeNet)](https://blog.csdn.net/wwt18811707971/article/details/127820299?spm=1001.2014.3001.5501)，此处不再列出。

其传输框图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/fa3cd74c126b47e69ea08cf42c095b3b.png#pic_center)

卷积层1数据传输如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/db825339199c47338ffa6259f3d84c95.png#pic_center)
其余层依此类似，只需类推。

## 2.2 补充
### 2.2.1 过拟合

根本原因是特征维度过多， **模型假设过于复杂， 参数过多， 训练数据过少**， 噪声过多，导致拟合的函数完美的预测训练集， 但对新数据的测试集预测结果差。 过度的拟合了训练数据， 而没有考虑到泛化能力。

过拟合主要受数据量和模型复杂度的影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f207555d1d264e88a92f0008318dca93.png#pic_center)


一句话就是：平时作业完成的非常好，但是考试就歇菜了。

### 2.2.2 Dropout

网络正向传播过程中随机失活一部分神经元，减少过拟合。Dropout 主要用在全连接层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe739a334c3f4fb6bb5e73f3f6be2815.png#pic_center)
# 3. demo 实现

## 3.1 数据集

本文使用花分类数据集，下载链接: [花分类数据集——http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

![在这里插入图片描述](https://img-blog.csdnimg.cn/527b7a338ae7489f910675254675615c.png#pic_center)

数据集划分参考这个[pytorch图像分类篇：3.搭建AlexNet并训练花分类数据集](https://blog.csdn.net/m0_37867091/article/details/107150142)

## 3.1 demo 结构：

![在这里插入图片描述](https://img-blog.csdnimg.cn/cf4ef19a778f4035aa038af45bff70af.png#pic_center)

在 CPU 训练的基础上，为了修改为 GPU 训练，因此单独修改了一个文件 train_gpu.py。

## 3.2 model.py

```python
"""
模型
"""


import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        """
        特征提取
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),   # 输入[3, 224, 224] 输出[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [48,27,27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),   # 输出 [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1), # 输出[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 输出[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   # 输出 [128, 6, 6]
        )
        """
        分类器
        """
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout 随机失活神经元, 比例诶0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    """
    权重初始化
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

"""
测试模型
"""
# if __name__ == '__main__':
#     input1 = torch.rand([224, 3, 224, 224])
#     model_x = AlexNet()
#     print(model_x)
    # output = AlexNet(input1)

```

此处的网络模型仅使用了一半的参数，即原 AlexNet 两块 GPU 中的一块。

### 3.2.1 nn.Sequential 介绍

nn.Sequential 是 nn.Module 的容器，用于**按顺序**包装一组网络层，在模型复杂情况下，使用 nn.Sequential 方法对模块划分。除了 nn.Sequetial，还有 nn.ModuleList 和 nn.ModuleDict。

![在这里插入图片描述](https://img-blog.csdnimg.cn/656d92f257e241348bc649a9e5ae901d.png#pic_center)

### 3.2.2  Tensor 展平

直接使用 torch.flatten 方法展平张量

```python 

 x = torch.flatten(x, start_dim=1)	# 二维平坦
 
 ```
 

## 3.3 model.py

### 3.3.1  导入包

```python

"""
训练(CPU)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm   # 显示进度条模块

from model import AlexNet

```

### 3.3.2 数据集预处理

```python

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),  # 随机裁剪, 再缩放为 224*224
                                    transforms.RandomHorizontalFlip(),  # 水平随机翻转
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
                                    transforms.Resize((224, 224)),  # 元组(224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

```

### 3.3.3 加载数据集

#### 3.3.3.1 读取数据路径

```python

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 读取数据路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
image_path = os.path.join(data_root, "data_set", "flower_data")
# image_path = data_root + "/data_set/flower_data/"
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

```

此处相比于 UP 主教程，修改了读取路径。

#### 3.3.3.2 加载训练集

```python

 train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"]
                                         )
 train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw
                                               )
```

#### 3.3.3.3 加载验证集

```python

val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"]
                                       )
val_num = len(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=nw
                                             )
```

#### 3.3.3.4 保存数据索引

```python

flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("calss_indices.json", 'w') as json_file:
        json_file.write(json_str)

```

### 3.3.4 训练过程

```python

net = AlexNet(num_classes=5, init_weights=True)     # 实例化网络(5分类)
    # net.to(device)
    net.to("cpu")   # 直接指定 cpu
    loss_function = nn.CrossEntropyLoss()   # 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=0.0002)     # 优化器(训练参数, 学习率)

    epochs = 10     # 训练轮数
    save_path = "./AlexNet.pth"
    best_accuracy = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()     # 开启Dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)     # 设置进度条图标
        for step, data in enumerate(train_bar):     # 遍历训练集,
            images, labels = data   # 获取训练集图像和标签
            optimizer.zero_grad()   # 清除历史梯度
            outputs = net(images)   # 正向传播
            loss = loss_function(outputs, labels)   # 计算损失值
            loss.backward()     # 方向传播
            optimizer.step()    # 更新优化器参数
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )
        # 验证
        net.eval()      # 关闭Dropout
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accuracy = acc / val_num
        print("[epoch %d ] train_loss: %3f    val_accurancy: %3f" %
              (epoch + 1, running_loss / train_steps, val_accuracy))
        if val_accuracy > best_accuracy:    # 保存准确率最高的
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), save_path)
    print("Finshed Training.")

```

训练过程可视化信息输出：

![在这里插入图片描述](https://img-blog.csdnimg.cn/fb6dcf0ecd424d6bb91480e680524068.png#pic_center)
**GPU 训练代码：** 仅在 CPU 训练的基础上做了数据转换处理。

```python 

"""
训练(GPU)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 读取数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    # image_path = data_root + "/data_set/flower_data/"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"]
                                         )
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("calss_indices.json", 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算
    nw = 0
    print(f"Using {nw} dataloader workers every process.")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw
                                               )
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"]
                                       )
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=nw
                                             )
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # test_data_iter = iter(val_loader)
    # test_image, test_label = next(test_data_iter)

    """ 测试数据集图片"""
    # def imshow(img):
    #     img = img / 2 + 0.5
    #     np_img = img.numpy()
    #     plt.imshow(np.transpose(np_img, (1, 2, 0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = "./AlexNet.pth"
    best_accuracy = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )
        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accuracy = acc / val_num
        print("[epoch %d ] train_loss: %3f    val_accurancy: %3f" %
              (epoch + 1, running_loss / train_steps, val_accuracy))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), save_path)
    print("Finshed Training.")


```

### 3.3.5 结果预测

```python 
"""
预测
"""

"""
预测
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_path = "./sunflowers01.jpg"
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)   # [N, C H, W]
    img = torch.unsqueeze(img, dim=0)   # 维度扩展
    # print(f"img={img}")
    json_path = "./calss_indices.json"
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # model = AlexNet(num_classes=5).to(device)   # GPU
    model = AlexNet(num_classes=5)  # CPU
    weights_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()    # 关闭 Dorpout
    with torch.no_grad():
        # output = torch.squeeze(model(img.to(device))).cpu()   #GPU
        output = torch.squeeze(model(img))      # 维度压缩
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy()))
        plt.show()


```

预测结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/d5c7d1622c404b7ba5f0112dd96bf93a.png#pic_center)

输入向日葵，预测准确率为 1.0 。

___

**欢迎关注公众号：【千艺千寻】，共同成长**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200704120239641.jpg#pic_center)



___
# 参考：
1. [pytorch图像分类篇：3.搭建AlexNet并训练花分类数据集](https://blog.csdn.net/m0_37867091/article/details/107150142)
2. [B站UP主——3.2 使用pytorch搭建AlexNet并训练花分类数据集](https://www.bilibili.com/video/BV1W7411T7qc/?spm_id_from=333.999.0.0&vd_source=103efe685ad4c1216c5d837f7dd7d25c)
3. [TensorFlow2学习十三、实现AlexNet](https://blog.51cto.com/u_4029519/5424126)
4. [PyTorch 11：模型容器与 AlexNet 构建](https://yey.world/2020/12/15/Pytorch-11/)
5. [PyTorch 10：模型创建步骤与 nn.Module](https://yey.world/2020/12/14/Pytorch-10/)
6. [卷积——动图演示](https://cs231n.github.io/assets/conv-demo/index.html)
7. [Python3 os.path() 模块](https://www.runoob.com/python3/python3-os-path.html)
