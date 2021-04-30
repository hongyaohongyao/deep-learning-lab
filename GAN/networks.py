import functools

import torch
from torch import nn


class DownSamplingResBlock(nn.Module):
    """
    下采样的残差块
    """

    def __init__(self, in_, out_, mid=-1):
        """
        :param in_:
        :param out_:
        :param mid: 中间通道数,非正数时和输出通道数相同
        """

        if mid <= 0:
            mid = out_
        super(DownSamplingResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_, mid, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid, out_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_),
        )
        self.extra = nn.Sequential() if in_ == out_ else nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(out_),
        )

    def forward(self, x):
        out = self.model(x)
        short_cut = self.extra(x)
        out = short_cut + out
        return out


class UpSamplingResBlock(nn.Module):
    """
    上采样的残差块
    """

    def __init__(self, in_, out_, mid=-1):
        """
        :param in_:
        :param out_:
        :param mid: 中间通道数,非正数时和输出通道数相同
        """
        if mid <= 0:
            mid = out_
        super(UpSamplingResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_, mid, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(mid, out_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_),
        )
        self.extra = nn.Sequential() if in_ == out_ else nn.Sequential(
            nn.ConvTranspose2d(in_, out_, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_),
        )

    def forward(self, x):
        out = self.model(x)
        short_cut = self.extra(x)
        out = short_cut + out
        return out


if __name__ == '__main__':
    UpSamplingResBlock(3, 2)


class ResNetGenerator(nn.Module):
    """
    残差结构的生成器
    """

    def __init__(self):
        super(ResNetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 32, kernel_size=7, padding=0),
                 nn.BatchNorm2d(32),
                 nn.ReLU(True)]
        # 下采样
        model += [DownSamplingResBlock(32, 64),
                  nn.LeakyReLU(0.2, True),
                  DownSamplingResBlock(64, 128),
                  nn.LeakyReLU(0.2, True)]
        # 上采样
        model += [UpSamplingResBlock(128, 64),
                  nn.LeakyReLU(0.2, True),
                  UpSamplingResBlock(64, 32),
                  nn.LeakyReLU(0.2, True)]
        # 输出
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, padding=0),
            nn.Tanh()  # 输出在[-1，1]
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim=64):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.model(x)


class ResNetGenerator2(nn.Module):
    """
    残差结构的生成器
    """

    def __init__(self):
        super(ResNetGenerator2, self).__init__()
        # 下采样
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, kernel_size=7, padding=0),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(True),
                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(True)]
        # 残差块
        for i in range(6):
            model += [ResBlock(256)]
        # 上采样
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        # 输出
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()  # 输出在[-1，1]
        ]
        self.model = nn.Sequential(*model)
        self.epoch = 0

    def forward(self, x):
        return self.model(x)


class PRDiscriminator(nn.Module):
    """
    PatchGAN Discriminator + ResNet
    """

    def __init__(self):
        super(PRDiscriminator, self).__init__()

        self.model = nn.Sequential(
            DownSamplingResBlock(3, 32),  # [b,3,256,256] -> [b,32,128,128]
            nn.LeakyReLU(0.2, True),
            DownSamplingResBlock(32, 64),  # [b,32,128,128] -> [b,64,64,64]
            nn.LeakyReLU(0.2, True),
            DownSamplingResBlock(64, 128),  # [b,64,64,64] -> [b,128,32,32]
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)  # [b,128,32,32] -> [b,1,32,32]
        )

    def forward(self, x):
        return self.model(x)


class PRDiscriminator2(nn.Module):
    """
    PatchGAN Discriminator + ResNet
    """

    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PRDiscriminator2, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class GANLoss(nn.Module):
    """
    判别器的损失函数
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # 声明常量
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        # 声明损失函数
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)  # 将目标常量扩展到判别结果等大小

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
