import torch
from torch import nn
from torch.utils.data import DataLoader

from ImageClassification.DigitsDataset import DigitsDataset
from ImageClassification.Util import draw_confusion_matrix, plot_img


class ResBlock(nn.Module):
    def __init__(self, in_, out_, mid=-1, stride=1):
        if mid <= 0:
            mid = out_
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_, mid, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(),
            nn.Conv2d(mid, out_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_),
        )
        self.extra = nn.Sequential() if in_ == out_ else nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_),
        )

    def forward(self, x):
        out = self.model(x)
        short_cut = self.extra(x)
        out = short_cut + out
        return out


class LikeResNet(nn.Module):
    def __init__(self):
        super(LikeResNet, self).__init__()
        self.model = nn.Sequential(
            ResBlock(4, 64, stride=2),
            nn.LeakyReLU(),
            ResBlock(64, 16, mid=128, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(12 * 12 * 16, 10)
        )
        self.epoch = 0

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    # 读取训练数据
    train_dataset = DigitsDataset()
    test_dataset = DigitsDataset(False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    model_path = "./like_res_net.pth"
    # model_net = LikeResNet()  # 创建模型
    # torch.save(model_net, model_path)  # 保存模型

    cuda_available = torch.cuda.is_available()  # 判断GPU是否存在
    device = torch.device("cuda:0" if cuda_available else "cpu")

    net = torch.load(model_path).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # 使用Adam优化器
    criterion = nn.CrossEntropyLoss().to(device)

    training_time = 100  # 训练次数
    last_epoch = net.epoch
    early_stop = 99.99
    for epoch in range(last_epoch + 1, last_epoch + 1 + training_time):
        # 先判断当前准确率，准确率大于85停止训练
        net.eval()
        test_loss = 0
        correct = 0
        batchNum = 0
        for X, y in test_loader:
            batchNum += 1
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
        # 训练
        net.train()
        for batchIdx, (X, y) in enumerate(train_loader):
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
        torch.save(net, model_path)

    net = torch.load(model_path).cpu()
    net.eval()
    # 查看前9张图的准确率
    X, y = next(iter(test_loader))
    out = net(X)
    pred = out.argmax(dim=1)
    plot_img(X, pred, "test")

    # 绘制混淆矩阵
    y_real = []
    y_pred = []
    for X, y in test_loader:
        out = net(X)
        y_real.extend(y.numpy())
        y_pred.extend(out.argmax(dim=1).numpy())
    draw_confusion_matrix(y_real, y_pred)


if __name__ == '__main__':
    main()
