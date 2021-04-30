import os
from random import Random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WafflesOrIceCreamDataset(Dataset):

    def __init__(self, is_train=True,  # 训练集还是测试集
                 resize=128,  # 设置图像大小
                 path='.\\dataset',
                 random_seed=23,  # 随机种子
                 training_data_ratio=0.8  # 训练集和测试集分割比例
                 ):
        # 获取文件路径
        ice_cream_path = os.path.join(path, 'ice_cream')
        imgs = [(os.path.join(ice_cream_path, file_name), 0) for file_name in os.listdir(ice_cream_path)]
        waffles_path = os.path.join(path, 'waffles')
        imgs.extend((os.path.join(waffles_path, file_name), 1) for file_name in os.listdir(waffles_path))
        # 打乱数据
        Random(random_seed).shuffle(imgs)
        # 分割数据集
        split_index = int(len(imgs) * training_data_ratio)
        self.imgs = imgs[:split_index] if is_train else imgs[split_index:]
        # 数据预处理
        self.transforms = transforms.Compose([
            lambda file_path: Image.open(file_path).convert('RGB'),
            transforms.Resize((int(resize * 1.25), int(resize * 1.25))),
            transforms.RandomRotation(15),  # 旋转一定角度
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将取值范围变换到[-1,+1]
        ])
        self.len = len(self.imgs)

    def __getitem__(self, index):
        file_path, label = self.imgs[index]
        img = self.transforms(file_path)
        return img, label

    def __len__(self):
        return self.len
