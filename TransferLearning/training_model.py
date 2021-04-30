import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from visdom import Visdom

from TransferLearning.WafflesOrIceCreamDataset import WafflesOrIceCreamDataset


def new_model(train_or_freeze=True, target_layer=None):
    if target_layer is None:
        target_layer = ['layer' + str(i) for i in range(1, 5)]
    resnet_model = resnet18(pretrained=True)
    # 冻结或解冻层
    for name, layer in resnet_model.named_children():
        if name in target_layer:
            layer.requires_grad_(train_or_freeze)
        else:
            layer.requires_grad_(not train_or_freeze)
    resnet_model.fc = nn.Linear(512, 2)
    return resnet_model


class TrainingModel:

    def __init__(self, start_epoch=-1, base_path=".\\checkpoints", new_model_func=None, env_name='0'):
        """
        :param start_epoch: 开始epoch
        :param base_path: 模型保存目录
        :param new_model_func: 新模型生成函数
        :param env_name: 环境名称
        """
        # 初始化参数
        self.epoch = start_epoch
        self.model_path = os.path.join(base_path, env_name)
        self.new_model = new_model_func
        # 创建模型保存目录
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        # 设置计算设备
        cuda_available = torch.cuda.is_available()  # 判断GPU是否存在
        self.device = torch.device("cuda:0" if cuda_available else "cpu")
        # 定义模型
        self._new_or_reload_model()
        self._reset_optim_n_criterion()
        # 开启visdom
        self.viz = Visdom(env='TransferLearning_%s' % env_name)
        if self.epoch < 0:
            self.viz.close()

    def _new_or_reload_model(self):
        # 是否重新开始训练
        if self.epoch < 0:
            self.net = self.new_model().to(self.device)  # 创建模型
            self.epoch = -1
        else:
            path = os.path.join(self.model_path, 'net_epoch_%d' % self.epoch)
            self.net = torch.load(path).to(self.device)

    def _save_model(self):
        path = os.path.join(self.model_path, 'net_epoch_%d' % self.epoch)
        torch.save(self.net, path)  # 保存模型

    def _reset_optim_n_criterion(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()),
            lr=1e-3)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def training(self,
                 early_stop=99.99,  # 提前停止的分数
                 training_time=100,  # 训练次数
                 batch_size=128,
                 print_freq=2,  # 训练过程中的打印频率
                 ):
        # 读取训练数据
        train_dataset = WafflesOrIceCreamDataset()
        test_dataset = WafflesOrIceCreamDataset(False)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        # 开始训练
        for epoch in range(self.epoch + 1, self.epoch + training_time + 1):
            # 测试分数
            self.net.eval()
            test_loss = 0
            correct = 0
            batchNum = 0
            for X, y in test_loader:
                batchNum += 1
                X, y = X.to(self.device), y.to(self.device)
                out = self.net(X)  # 前向传播
                test_loss += self.criterion(out, y).item()
                correct += out.argmax(dim=1).eq(y).sum().item()

            print("\nTest set: Last Epoch:{} Average loss:{:.4f}，Accuracy: {}/{} ({:.2f}%)\n"
                  .format(self.epoch,
                          test_loss / batchNum,
                          correct,
                          len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)
                          ))
            correct = 100. * correct / len(test_loader.dataset)
            self.visdom_correct(self.epoch, correct)
            if correct > early_stop:
                break
            # 训练
            batchNum = len(train_dataset) / batch_size
            self.net.train()
            for batchIdx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                out = self.net(X)  # 前向传播
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                if batchIdx % print_freq == 0:
                    print("Train Epoch:{} Batch:{} Average loss:{:.4f}"
                          .format(epoch,
                                  batchIdx,
                                  loss.item()
                                  ))
                epoch_process = round(epoch + batchIdx / batchNum, 2)
                self.visdom_loss(epoch_process, loss.item())
            self.epoch = epoch
            self._save_model()

    def visdom_correct(self, epoch, correct):
        self.viz.line([correct], [epoch], win='correct', update='append', opts=dict(
            title='correct',
            legend=['correct'],
            xlabel='epoch'
        ))

    def visdom_loss(self, epoch_process, loss):
        self.viz.line([loss], [epoch_process], win='loss', update='append', opts=dict(
            title='loss',
            legend=['loss'],
            xlabel='epoch'
        ))
