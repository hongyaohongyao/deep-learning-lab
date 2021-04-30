import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from GAN.Apple2OrangeDataset import Apple2OrangeDataset
from GAN.apple2orange_gan_model import Apple2OrangeGANModel


def main(test_path='./test', model_path="test_model", target_epoch=91):
    """
    :param model_path: 模型保存位置
    :param test_path: 转换结果保存的路径
    :param target_epoch: 第几个epoch的模型
    :return:
    """
    # 载入苹果图像训练集
    dataset_A = Apple2OrangeDataset(apple_or_orange=True, train_or_test=False)
    dataloader_A = DataLoader(dataset_A, batch_size=1)
    # 载入橘子图像训练集
    dataset_B = Apple2OrangeDataset(apple_or_orange=False, train_or_test=False)
    dataloader_B = DataLoader(dataset_B, batch_size=1)
    # 载入模型
    model = Apple2OrangeGANModel(is_train=True, model_path=model_path)
    model.load_models(target_epoch)
    model.eval()
    # 创建目录
    base_path = os.path.join(test_path, time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.isdir(base_path):
        os.makedirs(base_path)  # 创建生成所在目录
    path_A = os.path.join(base_path, 'apple2orange')
    # 创建苹果转橘子所在目录
    if not os.path.isdir(path_A):
        os.makedirs(path_A)
    # 创建橘子转苹果所在目录
    path_B = os.path.join(base_path, 'orange2apple')
    if not os.path.isdir(path_B):
        os.makedirs(path_B)
    with torch.no_grad():
        # 苹果转橘子
        for idx, real_A in enumerate(dataloader_A):
            real_A, fake_A = model.apple2orange(real_A)
            img = torch.cat([real_A, fake_A], dim=0) * 0.5 + 0.5
            file_name = os.path.join(path_A, 'apple2orange%s.png' % idx)
            save_img(file_name, img)
        # 橘子转苹果
        for idx, real_B in enumerate(dataloader_B):
            real_B, fake_B = model.orange2apple(real_B)
            img = torch.cat([real_B, fake_B], dim=0) * 0.5 + 0.5
            file_name = os.path.join(path_B, 'orange2apple%s.png' % idx)
            save_img(file_name, img)


def save_img(file_name, img):
    torchvision.utils.save_image(img, file_name, nrow=2)


if __name__ == '__main__':
    main()
