import json
import os

import numpy
import torch
import yaml

from nets.vit import VisionTransformer
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to "
                                          "RGBA images")

if __name__ == '__main__':
    # 配置文件路径
    config_yaml_file_path = os.path.join('model', 'vit_base_16', 'config.yaml')
    # 从配置文件加载配置
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # 模型相关
    image_size = config['model']['image_size']
    patch_size = config['model']['patch_size']
    embedding_dim = config['model']['embedding_dim']
    num_classes = config['model']['num_classes']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    pos_embedding_is_Parameter = config['model']['pos_embedding_is_Parameter']
    representation_size = config['model']['representation_size']
    k_dim = config['model']['k_dim']
    v_dim = config['model']['v_dim']
    ffn_hidden_size = config['model']['ffn_hidden_size']
    ffn_mode = config['model']['ffn_mode']
    drop_out_radio = config['model']['drop_out_radio']
    model_saved_path = config['train']['model_saved_path']
    model_saved_name = config['train']['model_saved_name']
    model_name = config['model']['model_name']
    is_norm_first = config['model']['is_norm_first']

    # 获取设备
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    if torch.cuda.is_available() is True:
        print(f'获取到设备 {torch.cuda.get_device_name(device.index)}')

    # 模型实例化
    if model_name == 'Vision Transformer':
        model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            pos_embedding_is_Parameter=pos_embedding_is_Parameter,
            representation_size=representation_size,
            k_dim=k_dim,
            v_dim=v_dim,
            ffn_hidden_size=ffn_hidden_size,
            ffn_mode=ffn_mode,
            is_norm_first=is_norm_first,
            drop_out_radio=drop_out_radio
        )
    else:
        raise ValueError(f'当前不支持模型{model_name}')

    # 模型权重加载
    if os.path.exists(os.path.join(model_saved_path, model_saved_name)):
        # 加载模型权重
        saved_state_dict = torch.load(str(os.path.join(model_saved_path, model_saved_name)))
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        model.load_state_dict(saved_state_dict)
    else:
        raise ValueError(f'模型权重文件{os.path.join(model_saved_path, model_saved_name)}不存在!')
    model = model.to(device)
    # print(f'模型结构如下：{model}')

    print('模型加载成功开始测试！')

    test_image_path = str(input('请输入要识别的图片的路径：'))
    # 读取图片
    test_image = Image.open(test_image_path).convert('RGB')
    # 图片预处理变换
    tf = transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 临时保存用做画图
    temp_image = numpy.array(test_image)
    # 图片预处理
    test_image = tf(test_image)
    # 添加batch_size维度
    test_image = test_image.unsqueeze(0)
    # 将图片转到对应设备
    test_image = test_image.to(device)

    # 读取label2name字典
    with open('rice_data/label2name.json', 'r') as f:
        label2name = json.load(f)
    print('正在识别...')
    # 模型转评估模式
    model.eval()
    # 无梯度环境下进行推理
    with torch.no_grad():
        output = model(test_image)
        # 获取预测结果index
        predict = torch.argmax(output, dim=-1)
        result = label2name[str(predict.item())]
    print(f'识别结果为：{result}')
    # 绘制图像
    plt.imshow(temp_image)
    # 自定义标题
    plt.title(result)
    # 显示图像
    plt.show()
