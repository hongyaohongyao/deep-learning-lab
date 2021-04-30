import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader
from visdom import Visdom

from LSTM.driving_alert_lstm import DrivingAlertLSTM
from LSTM.DrivingDataset import DrivingDataset


def get_metrics(y_real, y_pred):
    acc = accuracy_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_pred)
    return acc, recall, precision, auc


def main(new_model=False,
         title=None,
         num_layers=2,
         bidirectional=False,
         early_stop=87,
         training_time=50  # 训练次数
         ):
    test_metrics_title = 'test_metrics'
    test_loss_title = "test_loss"
    model_path = "./driving_alert_lstm"
    if title:
        test_metrics_title += '_' + title
        test_loss_title += '_' + title
        model_path += '_' + title
    model_path += '.pth'

    if new_model:
        model_net = DrivingAlertLSTM(num_layers=num_layers, bidirectional=bidirectional)  # 创建模型
        torch.save(model_net, model_path)  # 保存模型
    # 创建visdom窗口
    viz = Visdom(env='LSTM')
    if new_model:
        viz.close(test_metrics_title)
        viz.close(test_loss_title)

    if not (viz.win_exists(test_metrics_title)):
        viz.line([[0., 0., 0., 0.]], [0.],
                 win=test_metrics_title,
                 opts=dict(title=test_metrics_title,
                           legend=['accuracy', 'recall', 'precision', 'auc']))
    if not (viz.win_exists(test_loss_title)):
        viz.line([0.], [0.],
                 win=test_loss_title,
                 opts=dict(title=test_loss_title))
    # 读取训练数据
    train_dataset = DrivingDataset("./driving_train.csv")
    test_dataset = DrivingDataset("./driving_test.csv")

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128)

    cuda_available = torch.cuda.is_available()  # 判断GPU是否存在
    device = torch.device("cuda:0" if cuda_available else "cpu")

    net = torch.load(model_path).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # 使用Adam优化器
    criterion = torch.nn.MSELoss().to(device)

    last_epoch = net.epoch
    for epoch in range(last_epoch + 1, last_epoch + 1 + training_time):
        # 先判断当前准确率，准确率大于early_stop停止训练
        test_loss = 0
        net.eval()
        y_real = []
        y_pred = []
        batchNum = 0
        with torch.no_grad():
            for X, y in test_loader:
                batchNum += 1
                X, y = X.to(device), y.to(device)
                out = net(X)  # 前向传播
                test_loss += criterion(out, y).item()
                y_real.extend(y.cpu().detach().numpy())
                y_pred.extend(torch.round(out).cpu().detach().numpy())
            acc, recall, precision, auc = get_metrics(y_real, y_pred)
        print("\nTest set: Last Epoch:{} Average loss:{:.4f}，Accuracy: {:.2f}% AUC: {:.2f}%\n"
              .format(net.epoch,
                      test_loss / batchNum,
                      100 * acc,
                      100 * auc
                      ))
        viz.line([[acc, recall, precision, auc]], [epoch],
                 win=test_metrics_title, update='append')
        viz.line([test_loss / batchNum], [epoch],
                 win=test_loss_title,
                 update='append')
        if 100 * acc > early_stop:
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

            if batchIdx % 100 == 0:
                print("Train Epoch:{} Batch:{} Average loss:{:.4f}"
                      .format(epoch,
                              batchIdx,
                              loss.item()
                              ))
        net.epoch = epoch
        torch.save(net, model_path)

    # net = torch.load(model_path).cpu()
    # net.eval()


if __name__ == '__main__':
    # main(new_model=True)
    # main(new_model=True, title='layer3', num_layers=3)
    main(new_model=True, title='Bi', bidirectional=True)
