# Vision Transformer从零重构-基于Pytorch-水稻病虫害数据集预训练
作者: 范昊

author: Hao Fan

本仓库仅供学习交流使用，实现参考 [Vision Transformer](https://arxiv.org/abs/2010.11929) 和 [Pytorch](https://github.com/pytorch/vision) 。

## 简介
- `模型`: 本仓库从MultiHeadAttention开始一步步重构了Vit，相比官方实现更加通俗易懂，并且补充了大量注释；
- `数据集`: 自定义的水稻病虫害多分类数据集，也可以很轻松地替换为自己的多分类数据集，仅需将每个类别的图片放入同一个文件夹， 
然后将所有类别的文件夹放入rice_data（可根据自己的数据集名称进行修改）即可；
- `模型训练和模型保存逻辑`: 本仓库实现了基础的模型训练、日志和模型权重保存逻辑；

---

<u> 项目结构介绍 </u>

- `utils.py`: 一些工具函数等，当前包含一个设置日志器的函数；
- `callbacks.py`: 用于记录训练日志的类定义；
- `dataset.py`: 数据集加载定义，内部定义了DataSet类，只需传入多分类数据集文件路径和数据划分即可使用（具体类名可按自己需求进行更改）；
- `train.py`: 训练脚本，仅需修改配置文件路径即可运行并开始训练模型；
- `test.py`: 测试脚本，可以选择单张图像进行测试；
- `📁 nets`: 存放模型网络结构:
  - `vit.py`: 本仓库实现的Vision Transformer，从ScaledProduct到MultiHeadAttention再到Encoder一步步重构；
- `📁 model`: 模型配置定义、训练日志以及保存权重等：
  - `📁 vit_base_16`: 重构vit-base-16模型的配置文件：
    - `📁 logs`: 训练日志文件：
      - `📁 loss_and_acc_{time}`: {time}时刻开始训练的日志，包含accuracy和loss等的记录；
    - `📁 weights`: 存放权重文件pth，由于github仓库上传限制，已将在水稻病虫害多分类数据集上预训练的权重上传至 [百度网盘](https://pan.baidu.com/share/init?surl=Hub7i6wy4_s0-9fLbs7OHw&pwd=elw8)， 请下载后放入本文件夹；
  - `config.yaml`: 模型配置文件，包括模型定义超参和训练超参以及数据集路径等；
- `📁 rice_data`: 水稻病虫害数据集，包含各个类别的文件夹以及映射json（之后可以替换为自己的多分类数据集）：
  - `label2name.json`: 类别索引标签和类别名的对应json字典；
  - `...📁 类别名`: 每个类别的图片，由于数据集版权问题，这里只能上传少部分图片以供模型测试；


## Usage
### 模型
主要用法是使用`VisionTransformer`模块（[vit.py](nets/vit.py)），实例化完成后，仅需传入shape为`[B, C, H, W]`的图像数据即可，可用以下代码进行测试：
```python
import torch
from nets.vit import VisionTransformer
data = torch.randn(8, 3, 224, 224)
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    embedding_dim=768,
    num_classes=10,
    num_heads=8,
    num_layers=6)
print(model(data).shape)
```
对于模型配置部分，如需自定义模型超参进行训练，可以通过在`📁 model`中新建模型文件夹，并按照 <u>项目结构介绍</u> 中的文件结构新建`📁 logs`和`📁 weights`文件夹
（`📁 loss_and_acc_{time}`不用新建，会自动生成），然后再按照`config.yaml`文件中定义的进行配置即可（[config.yaml](model/vit_base_16/config.yaml)）。

### 数据集加载
另外多分类数据集部分主要是使用`MultiClassDataSet`模块（[dataset.py](dataset.py)），仅需在实例化的时候传入数据集路径以及数据集切分要求即可，可用以下代码进行测试：
```python
from dataset import MultiClassDataSet
from torch.utils.data import DataLoader
# 数据集加载路径
file_path_test = r'你的数据集加载路径'
batch_size = 8
train_dataset = MultiClassDataSet(file_path=file_path_test,
                                  resize=224,
                                  mode='train')
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
# 用DataLoader进行一轮批加载测试
for images, labels in train_dataloader:
    print(images.shape)
    break
```

### 训练脚本运行
无需多言，修改完`train.py`（[train.py](train.py)）中的
```python
# 配置文件路径
config_yaml_file_path = r'你的配置文件路径'
```
直接运行即可：
```shell
python train.py
```

### 测试脚本运行
和训练脚本类似，修改完`test.py`（[test.py](test.py)）中的
```python
# 配置文件路径
config_yaml_file_path = r'你的配置文件路径'
```
直接运行：
```shell
python test.py
```
之后终端输入要测试的图片路径即可（可以直接拿rice_data中的采样数据测试）。
