import os

import torch
import yaml
from tqdm import tqdm

from nets.vit import VisionTransformer, vit_pytorch
from dataset import MultiClassDataSet
from utils import setting_logging
from torch.utils.data import DataLoader
from callbacks import LossAndAccuracyHistory
from torchvision.models import vision_transformer, vit_b_16
import warnings

warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to "
                                          "RGBA images")
# 限制可用显卡编号，如需多卡请将要添加的显卡编号（可用nvidia-smi查看）添加入字符串，用半角逗号分隔
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # 配置文件路径
    config_yaml_file_path = os.path.join('model', 'vit_base_16_pytorch', 'config.yaml')
    # 从配置文件加载配置
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    logger_name = config['train']['logger_name']
    # 设置日志器
    logger = setting_logging(logger_name)
    # 总的迭代次数
    Epochs = config['train']['Epochs']
    # 批处理大小
    batch_size = config['train']['batch_size']

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
    save_log_dir = config['train']['save_log_dir']
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    dataset_path = config['train']['dataset_path']
    is_norm_first = config['model']['is_norm_first']

    # 获取训练设备
    device = (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    if torch.cuda.is_available() is True:
        logger.info(f'获取到设备 {torch.cuda.get_device_name(device.index)}')

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
    elif model_name == 'Vision Transformer Pytorch':
        model = vit_pytorch(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_hidden_dim=ffn_hidden_size,
            drop_out_radio=drop_out_radio)
    else:
        raise ValueError(f'当前不支持模型{model_name}')

    # 模型权重加载
    if os.path.exists(os.path.join(model_saved_path, model_saved_name)):
        # 加载模型权重
        saved_state_dict = torch.load(str(os.path.join(model_saved_path, model_saved_name)))
        min_val_loss = saved_state_dict['min_val_loss']
        saved_state_dict.pop('min_val_loss')
        # 通过自定义加载过程匹配键
        new_model_state_dict = {}
        for key, value in saved_state_dict.items():
            # 修改键的方式以匹配新模型的结构
            new_key = key.replace("module.", "")
            new_model_state_dict[new_key] = value
        model.load_state_dict(new_model_state_dict)
    else:
        min_val_loss = 1e100
    model = model.to(device)

    # 支持和不支持gpu分两种情况处理，因为DataParallel没有GPU会报错
    if torch.cuda.is_available():
        # 指定使用的GPU编号(为空则是可用的全部)
        device_ids = None
        model = torch.nn.DataParallel(module=model, device_ids=device_ids)

    logger.info(f'模型结构如下：{model}')

    # 实例化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 实例化损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 实例化训练集和验证集
    train_dataset = MultiClassDataSet(file_path=dataset_path,
                                      resize=image_size)
    val_dataset = MultiClassDataSet(file_path=dataset_path,
                                    resize=image_size,
                                    mode='val')
    # 实例化数据加载器
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    # 训练损失和准确度记录
    loss_history = LossAndAccuracyHistory(log_dir=save_log_dir,
                                          model=model,
                                          input_shape=[batch_size, 3, image_size, image_size],
                                          device=device)
    # 计算每一个世代的长度
    epoch_step = len(train_dataset) // batch_size
    epoch_step_val = len(val_dataset) // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    logger.info('模型加载成功开始训练！')
    for epoch in range(Epochs):

        # 训练集总损失
        total_train_loss = 0.0
        # 训练集总准确度
        total_train_acc = 0.0
        # 验证集总损失
        total_val_loss = 0.0
        # 验证集总准确度
        total_val_acc = 0.0
        # 模型转训练模式
        model.train()
        logger.info(f"第{epoch + 1}轮开始训练")
        # 实例化进度条
        pbar = tqdm(total=epoch_step,
                    desc=f'Epoch {epoch + 1}/{Epochs}',
                    postfix={'train_loss': '?',
                             'train_avg_loss': '?',
                             'train_acc': '?',
                             'train_avg_acc': '?'}, mininterval=0.3)
        # 迭代训练集数据进行训练
        for idx, (images, labels) in enumerate(train_dataloader):
            # 最后数据不足的batch直接去掉
            if idx >= epoch_step:
                break
            # 把数据转移到device上
            images = images.to(device)
            labels = labels.to(device)
            # [batch_size, channels, h, w] -> [batch_size, num_classes]
            out = model(images)
            # [batch_size, num_classes] -> [batch_size]
            predict = torch.argmax(out, dim=-1)

            # 清空上一轮的梯度信息
            optimizer.zero_grad()
            # 计算损失
            loss = criterion(out, labels)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 计算误差和精确度
            with torch.no_grad():
                # 计算比较预测正确的数
                equal = torch.eq(predict, labels)
                # 计算准确度
                accuracy = torch.mean(equal.float())
            total_train_loss += loss.item()
            total_train_acc += accuracy.item()

            pbar.set_postfix(**{'train_loss': loss.item(),
                                'train_avg_loss': total_train_loss / (idx + 1),
                                'train_acc': accuracy.item(),
                                'train_avg_acc': total_train_acc / (idx + 1)})
            pbar.update(1)
        pbar.close()
        logger.info(f'第{epoch + 1}轮训练结束')

        # 把模型切换成验证模式
        model.eval()
        logger.info(f"第{epoch + 1}轮开始验证")
        # 实例化进度条
        pbar = tqdm(total=epoch_step_val,
                    desc=f'Epoch {epoch + 1}/{Epochs}',
                    postfix={'val_loss': '?',
                             'val_avg_loss': '?',
                             'val_acc': '?',
                             'val_avg_acc': '?'}, mininterval=0.3)
        # 使用无梯度环境，禁止梯度图更新
        with torch.no_grad():
            # 迭代训练集数据进行训练
            for idx, (images, labels) in enumerate(val_dataloader):
                # 最后数据不足的batch直接去掉
                if idx >= epoch_step_val:
                    break
                # 把数据转移到device上
                images = images.to(device)
                labels = labels.to(device)
                # [batch_size, channels, h, w] -> [batch_size, num_classes]
                out = model(images)
                # [batch_size, num_classes] -> [batch_size]
                predict = torch.argmax(out, dim=-1)

                # 计算损失
                loss = criterion(out, labels)
                # 计算误差和精确度
                # 计算比较预测正确的数
                equal = torch.eq(predict, labels)
                # 计算准确度
                accuracy = torch.mean(equal.float())

                total_val_loss += loss.item()
                total_val_acc += accuracy.item()

                pbar.set_postfix(**{'val_loss': loss.item(),
                                    'val_avg_loss': total_val_loss / (idx + 1),
                                    'val_acc': accuracy.item(),
                                    'val_avg_acc': total_val_acc / (idx + 1)})
                pbar.update(1)

        pbar.close()
        logger.info(f'第{epoch + 1}轮验证结束')
        # 保存训练日志
        loss_history.append_loss(epoch + 1, total_train_loss / epoch_step, total_val_loss / epoch_step_val)
        loss_history.append_acc(epoch + 1, total_train_acc / epoch_step, total_val_acc / epoch_step_val)
        if total_val_loss / epoch_step_val < min_val_loss:
            logger.info(f'将最好的模型保存到{model_saved_path}')
            min_val_loss = total_val_loss / epoch_step_val
            model_dict = model.state_dict()
            model_dict['min_val_loss'] = total_val_loss / epoch_step_val
            torch.save(model_dict, str(os.path.join(model_saved_path, model_saved_name)))
        # torch.save(model.state_dict(), os.path.join(model_saved_path, 'last_epoch_model.pth'))
