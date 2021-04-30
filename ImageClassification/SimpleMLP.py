import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ImageClassification.DigitsDataset import DigitsDataset
from ImageClassification.Util import plot_img, draw_confusion_matrix


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.L1 = nn.Linear(100 * 100 * 4, 256)
        self.L2 = nn.Linear(256, 64)
        self.L3 = nn.Linear(64, 10)
        self.epoch = 0

    def forward(self, x):
        x = F.leaky_relu(self.L1(x))
        x = F.leaky_relu(self.L2(x))
        x = self.L3(x)
        return x


def main():
    # 读取训练数据
    train_dataset = DigitsDataset()
    test_dataset = DigitsDataset(False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    # simple_mlp = SimpleMLP()  # 创建模型
    # torch.save(simple_mlp, "./simple_mlp.pth")  # 保存模型

    cuda_available = torch.cuda.is_available()  # 判断GPU是否存在
    device = torch.device("cuda:0" if cuda_available else "cpu")

    net = torch.load("./simple_mlp.pth").to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # 使用Adam优化器
    criterion = nn.CrossEntropyLoss().to(device)

    training_time = 100  # 训练次数
    last_epoch = net.epoch
    early_stop = 93.6
    for epoch in range(last_epoch + 1, last_epoch + 1 + training_time):
        # 先判断当前准确率，准确率大于85停止训练
        test_loss = 0
        correct = 0
        net.eval()
        batchNum = 0
        for X, y in test_loader:
            batchNum += 1
            X = X.view(X.size(0), 100 * 100 * 4)  # 将图像打平
            X, y = X.to(device), y.to(device)
            out = net(X)  # 前向传播
            test_loss += criterion(out, y).item()
            correct += out.argmax(dim=1).eq(y).sum().item()

        print("\nTest set: Last Epoch:{} Average loss:{:.4f}，Accuracy: {}/{} ({:.2f}%)\n"
              .format(net.epoch,
                      test_loss / batchNum,
                      correct,
                      len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)
                      ))
        if 100. * correct / len(test_loader.dataset) > early_stop:
            break
        net.train()
        for batchIdx, (X, y) in enumerate(train_loader):
            X = X.view(X.size(0), 100 * 100 * 4)  # 将图像打平
            X, y = X.to(device), y.to(device)
            out = net(X)  # 前向传播
            loss = criterion(out, y)

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batchIdx % 10 == 0:
                print("Train Epoch:{} Batch:{} Average loss:{:.4f}"
                      .format(epoch,
                              batchIdx,
                              loss.item()
                              ))
        net.epoch = epoch
        torch.save(net, "./simple_mlp.pth")

    net = torch.load("./simple_mlp.pth").cpu()
    net.eval()
    # 查看前9张图的准确率
    X, y = next(iter(test_loader))
    out = net(X.view(X.size(0), 100 * 100 * 4))
    pred = out.argmax(dim=1)
    plot_img(X, pred, "test")

    # 绘制混淆矩阵
    y_real = []
    y_pred = []
    for X, y in test_loader:
        X = X.view(X.size(0), 100 * 100 * 4)  # 将图像打平
        out = net(X)
        y_real.extend(y.numpy())
        y_pred.extend(out.argmax(dim=1).numpy())
    draw_confusion_matrix(y_real, y_pred)


if __name__ == '__main__':
    main()
