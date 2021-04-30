import itertools
import os

import torch

from GAN import networks
from GAN.networks import ResNetGenerator, PRDiscriminator


def set_requires_grad(nets, requires_grad=False):
    """
    设置是否梯度计算
    :param nets:
    :param requires_grad:
    :return:
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Apple2OrangeGANModel:
    def __init__(self, is_train=True, model_path='models'):
        """
        初始化
        """
        # 一般参数
        self.model_path = model_path
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)  # 创建模型所在目录
        self.is_train = is_train
        self.epoch = 0
        self.model_names = ['G_A', 'G_B']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # 硬件加速
        cuda_available = torch.cuda.is_available()  # 判断GPU是否存在
        self.device = torch.device("cuda:0" if cuda_available else "cpu")
        # 定义生成器
        self.netG_A = ResNetGenerator().to(self.device)
        self.netG_B = ResNetGenerator().to(self.device)

        # 训练所需参数
        if is_train:
            self.model_names += ['D_A', 'D_B']
            # 定义判别器
            self.netD_A = PRDiscriminator().to(self.device)
            self.netD_B = PRDiscriminator().to(self.device)
            # 定义损失函数
            self.criterionGAN = networks.GANLoss().to(self.device)  # real_img->netG_AB->netG_BA->fake_img
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # 定义优化器B
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))

    def _backward_D_basic(self, netD, real, fake):
        """
        :param netD: 判别器
        :param real: 原图
        :param fake: 生成图
        :return: 损失
        """
        # 真图片
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # 假图片
        pred_fake = netD(fake.detach())  # fake图片连接生成器，因此需要阻断反向传播
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # 合并损失并计算梯度
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def forward(self, real_A, real_B):
        """
        前向传播
        :param real_A: 真图片A
        :param real_B: 真图片B
        :return:
        """
        # real_A->[G_A]->fake_B->[G_B]->rec_A
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)
        # real_B->[G_B]->fake_A->[G_A]->rec_B
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)
        return fake_A, fake_B, rec_A, rec_B

    def _backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """
        训练生成器
        :param real_A: 原图A
        :param real_B: 原图B
        :param fake_A: 生成图A
        :param fake_B: 生成图B
        :param rec_A: 循环生成图A
        :param rec_B: 循环生成图B
        :return: 损失计算中间值的字典
        """
        lambda_idt = 0.5
        lambda_A = 1
        lambda_B = 1
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.netG_A(real_B)
            loss_idt_A = (self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt).item()
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.netG_B(real_A)
            loss_idt_B = (self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt).item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        return {'loss_G': loss_G.item(), 'loss_G_A': loss_G_A.item(), 'loss_G_B': loss_G_B.item(),
                'loss_cycle_A': loss_cycle_A.item(), 'loss_cycle_B': loss_cycle_B.item(),
                'loss_idt_A': loss_idt_A, 'loss_idt_B': loss_idt_B}

    def train_a_batch(self, real_A, real_B):
        """
        训练一批数据
        :param real_A: 真的苹果图片
        :param real_B: 真的橘子图片
        :return:
        """
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        if not self.is_train:
            return
        # ===训练生成器===
        # 前向传播
        fake_A, fake_B, rec_A, rec_B = self.forward(real_A, real_B)
        # 反向传播
        set_requires_grad([self.netD_A, self.netD_B], False)  # 冻结判别器参数
        self.optimizer_G.zero_grad()  # 梯度清零
        args_dict = self._backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)  # 计算梯度
        self.optimizer_G.step()  # 更新生成器权重参数

        # ===训练判别器===
        set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # 清空判别器D_A和D_B的梯度
        loss_D_A = self._backward_D_basic(self.netD_A, real_B, fake_B)  # 计算D_A的梯度
        loss_D_B = self._backward_D_basic(self.netD_B, real_A, fake_A)  # 计算D_B的梯度
        self.optimizer_D.step()  # 更新判别器权重参数
        # 将损失和图像加入字典
        args_dict['loss_D_A'] = loss_D_A.item()
        args_dict['loss_D_B'] = loss_D_B.item()
        args_dict['real_A'] = real_A
        args_dict['real_B'] = real_B
        args_dict['fake_A'] = fake_A
        args_dict['fake_B'] = fake_B
        args_dict['rec_A'] = rec_A
        args_dict['rec_B'] = rec_B
        return args_dict

    def save_models(self, epoch):
        """
        保存模型
        """
        if not self.is_train:
            return
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.model_path, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)

    def load_models(self, epoch):
        """
        载入模型
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.model_path, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = torch.load(load_path, map_location=self.device)
                net.load_state_dict(state_dict)

    def train(self, is_train=True):
        """设置模型的训练状态"""
        self.is_train = is_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train(is_train)

    def eval(self):
        """设置模型为测试状态"""
        self.train(False)

    def apple2orange(self, real_A):
        if self.is_train:
            return
        real_A = real_A.to(self.device)
        return real_A, self.netG_A(real_A)

    def orange2apple(self, real_B):
        if self.is_train:
            return
        real_B = real_B.to(self.device)
        return real_B, self.netG_B(real_B)

    def target_loss(self, real_img):
        """
        用于判断模型是否合适的方法，目标损失默认是loss_GA
        :param real_img:
        :return:
        """
        real_img = real_img.to(self.device)
        # real_A->[G_A]->fake_B->[D_A]->[loss]
        with torch.no_grad():
            return self.criterionGAN(self.netD_A(self.netG_A(real_img)), True).item()
