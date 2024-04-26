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
  