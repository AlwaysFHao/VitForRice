import csv
import glob
import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import tensor


class MultiClassDataSet(Dataset):
    r"""
    自定义数据集
    """

    def __init__(self, file_path: str, resize: int, mode: str = 'train'):
        """
        初始化，读取数据集
        :param file_path: 数据集路径
        :param resize: 需要裁剪大小
        :param mode: 训练集or测试集 输入 【'train' or 'val' or 'test'】
        """
        super(MultiClassDataSet, self).__init__()
        # 数据集路径
        self.file_path = file_path
        # 图片裁剪大小
        self.resize = resize
        # 做个体名到编码的映射关系
        self.name2label = {}  # 'str':0
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.8),  # 随机透视变换
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((int(self.resize), int(self.resize))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # 遍历读取个体名，注意这里需要排序一下，因为listdir返回顺序不固定
        for name in sorted(os.listdir((os.path.join(file_path)))):
            # 判断是否为文件夹，如果不是文件夹则直接忽略
            if not os.path.isdir(os.path.join(file_path, name)):
                continue
            # 字典的keys方法返回当前字典中存在的所有可用key值列表，取列表长度后，按照个体名扫描顺序给个体进行顺序编码
            self.name2label[name] = len(self.name2label.keys())

        # 保存类别编码字典
        with open(os.path.join(file_path, "label2name.json"), "w") as json_file:
            self.label2name = {v: k for k, v in self.name2label.items()}
            json.dump(self.label2name, json_file)

        # 读取数据集的图片标签csv
        self.images, self.labels = self.load_csv('images.csv')
        # 判断数据集模式
        if mode == 'train':
            # 训练集读取60%数据
            self.images = self.images[:int(0.70 * len(self.images))]
            self.labels = self.labels[:int(0.70 * len(self.labels))]
        elif mode == 'val':
            # 验证集读取20%数据
            self.images = self.images[int(0.70 * len(self.images)):int(0.85 * len(self.images))]
            self.labels = self.labels[int(0.70 * len(self.labels)):int(0.85 * len(self.labels))]
        elif mode == 'test':
            # 测试集读取20%数据
            self.images = self.images[int(0.85 * len(self.images)):]
            self.labels = self.labels[int(0.85 * len(self.labels)):]
        else:
            raise ValueError("请输入正确的读取模式！\a【'train' or 'val' or 'test'】")

    def load_csv(self, filename: str) -> (list, list):
        """
        读取数据集标签csv文件
        :param filename: csv标签文件名
        :return: 图片路径url列表和标签列表
        """
        # 判断是否存在csv文件，不存在则创建
        if not os.path.exists(os.path.join(self.file_path, filename)):
            # 保存要读取图片的列表
            images: list = []
            # 读取字典中的key值，组合文件路径
            for name in self.name2label.keys():
                # 使用glob模块去扫描出所有的png、jpg和jpeg图片
                # os.path.join方法可以在不同系统下都成功拼接文件路径
                images += glob.glob(os.path.join(self.file_path, name, '*.png'))
                images += glob.glob(os.path.join(self.file_path, name, '*.jpg'))
                images += glob.glob(os.path.join(self.file_path, name, '*.jpeg'))
            # 随机打乱
            random.shuffle(images)
            # 将图片路径和标签写入csv文件
            with open(os.path.join(self.file_path, filename), mode='w', newline='') as file:
                writer = csv.writer(file)
                for img in images:
                    # 按照系统路径分级符分割文件名，读取出个体名
                    name = img.split(os.sep)[-2]
                    # 读取出对应的编码
                    label = self.name2label[name]
                    # 写入csv
                    writer.writerow([img, label])

        # 存储图片路径和标签
        images, labels = [], []
        # 读取csv文件
        with open(os.path.join(self.file_path, filename)) as file:
            reader = csv.reader(file)
            # 按行读取
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels), "图片和标签长度不匹配"
        return images, labels

    def __len__(self) -> int:
        """
        获取数据集长度
        :return:
        """
        return len(self.images)

    @staticmethod
    def de_normalize(x_hat):
        """
        反归一化方法
        :param x_hat:
        :return:
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, item) -> (tensor, tensor):
        """
        读取index位置数据
        :param item: 数据集索引
        :return: 数据集图片和标签
        """
        img, label = self.images[item], self.labels[item]
        # 打开图像
        img = Image.open(img).convert('RGB')
        # 数据增强
        img = self.tf(img)
        # label转tensor
        label = torch.tensor(label)

        return img, label


if __name__ == '__main__':
    file_path_test = ''
    batch_size = 8
    train_dataset = MultiClassDataSet(file_path=file_path_test,
                                      resize=224,
                                      mode='train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    for images, labels in train_dataloader:
        print(images.shape)
