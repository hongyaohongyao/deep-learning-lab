import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from TransferLearning.WafflesOrIceCreamDataset import WafflesOrIceCreamDataset


def print_resnet18():
    """
    查看resnet18的网络结构
    """
    resnet_model = resnet18(pretrained=True)
    print([name for name, m in resnet_model.named_children()])
    print(resnet_model)


def feature_extraction_img(base_path=".\\conv"):
    """
    生成resnet18卷积层特征提取的结果，保存到./conv/文件夹下
    """
    # 创建目录
    if not os.path.isdir(base_path):
        os.makedirs(base_path)  # 创建生成所在目录

    dataset = WafflesOrIceCreamDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    resnet_model = resnet18(pretrained=True).eval()

    c = 0
    for m in resnet_model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        m.register_forward_hook(lambda mo, i_, o, conv_n=c: draw_conv_output_plot(o, conv_n, base_path))
        c += 1
    # 训练一张图片
    with torch.no_grad():
        for x, i in dataloader:
            resnet_model(x)
            break


def draw_conv_output_plot(o, conv_n, base_path):
    """
    打印卷积层的输出结果
    :param o: 输出
    :param conv_n: 第几个卷积层
    :param base_path: 输出目录
    """
    o = o[0, :64].detach() * 0.5 + 0.5
    o = torch.unsqueeze(o, 1)  # BWH=>BCWH
    path = os.path.join(base_path, 'conv%d.png' % conv_n)
    torchvision.utils.save_image(o, path)


if __name__ == '__main__':
    print_resnet18()
    feature_extraction_img()
