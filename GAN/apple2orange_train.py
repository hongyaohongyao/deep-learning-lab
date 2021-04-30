from torch.utils.data import DataLoader
from visdom import Visdom

from GAN.Apple2OrangeDataset import Apple2OrangeDataset
from GAN.apple2orange_gan_model import Apple2OrangeGANModel


def main(start_epoch=0, batch_size=5, from_last=False, env_name='apple2orangeGAN6'):
    """

    :param env_name: visdom 环境名称
    :param from_last: 从上一个模型的结果开始
    :param start_epoch:  开始训练的epoch，非正整数表示重新开始训练
    :param batch_size:
    :return:
    """
    # 载入苹果图像训练集
    dataset_A = Apple2OrangeDataset(apple_or_orange=True, train_or_test=True)
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    # 载入橘子图像训练集
    dataset_B = Apple2OrangeDataset(apple_or_orange=False, train_or_test=True)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)
    # 载入模型
    model = Apple2OrangeGANModel(is_train=True)
    # 训练参数
    train_times = 100  # 训练train_times次停止
    # 训练初始化
    viz = Visdom(env=env_name)
    # 新的训练或从之前的训练开始
    model_epoch = (start_epoch - 1) if from_last else start_epoch
    if model_epoch < 0:
        viz.close()
        start_epoch = 0
    else:
        # 载入模型参数
        model.load_models(model_epoch)
    # 开始训练
    for epoch in range(start_epoch, start_epoch + train_times):
        combined_dataloader = zip(dataloader_A, dataloader_B)
        batchNum = min(len(dataset_A), len(dataset_B)) / batch_size
        min_loss = 9999
        for batchIdx, (real_A, real_B) in enumerate(combined_dataloader):
            if batchIdx % 5 == 0:
                # 保存模型
                cur_loss = model.target_loss(real_A)
                if cur_loss < min_loss:
                    model.save_models(epoch)
                    min_loss = cur_loss
                    print("epoch: %d, saved as new loss_G_A=%.3f" % (epoch, cur_loss))
            args_dict = model.train_a_batch(real_A, real_B)
            if batchIdx % 5 == 0:
                process = batchIdx / batchNum
                plot_losses(viz, 'losses', round(epoch + process, 5), args_dict)
                print_losses(epoch, process, args_dict)
                # 绘制图像
                plot_img(viz, 'real_A', '真实图像A', args_dict)
                plot_img(viz, 'fake_B', '生成图像B', args_dict)
                plot_img(viz, 'rec_A', '还原图像A', args_dict)
                plot_img(viz, 'real_B', '真实图像B', args_dict)
                plot_img(viz, 'fake_A', '生成图像A', args_dict)
                plot_img(viz, 'rec_B', '还原图像B', args_dict)
        # 尝试保存模型
        cur_loss = model.target_loss(real_A)
        if cur_loss < min_loss:
            model.save_models(epoch)
            print("epoch: %d, saved as new loss_G_A=%.3f" % (epoch, cur_loss))
        print("\nepoch: %d, finished \n" % epoch)


loss_names = ['loss_' + name for name in ['G', 'D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']]


def plot_losses(visdom, title, epoch, args_dict):
    """
    展示损失图
    :param visdom: visdom对象
    :param title: 窗口标题
    :param epoch: 横坐标表示模型经过几次完整的训练
    :param args_dict: 保存训练过程参数的字典
    """
    visdom.line([[args_dict[name] for name in loss_names]], [epoch],
                win=title,
                update='append',
                opts=dict(
                    title=title,
                    legend=loss_names
                ))


def plot_img(visdom, pic_name, caption, args_dict):
    """
    绘制当前生成器生成的图像
    :param pic_name: 图片在args_dict中的名称
    :param caption: 说明文字
    :param visdom: visdom对象
    :param args_dict: 保存训练过程参数的字典
    """
    imgs = args_dict[pic_name]
    imgs = 255 * (imgs * 0.5 + 0.5)
    # 绘制图像
    visdom.images(imgs,
                  win=pic_name,
                  opts=dict(
                      caption=caption,
                  ))


def print_losses(epoch, epoch_process, args_dict):
    """
    :param epoch_process: 本次epoch 完成的进度
    :param epoch: 模型在第几次完整的训练
    :param args_dict: 保存训练过程参数的字典
    """
    message = '(epoch: %d, process: %.2f%%) ' % (epoch, 100 * epoch_process)
    for name in loss_names:
        message += '%s: %.3f ' % (name, args_dict[name])
    print(message)


if __name__ == '__main__':
    main()
