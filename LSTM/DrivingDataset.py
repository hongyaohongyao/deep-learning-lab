import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrivingDataset(Dataset):

    def __init__(self, path, seq_num=10, preprocessor_path="./driving_data_preprocessor.skl"):
        self.seq_num = seq_num
        preprocessor = joblib.load(preprocessor_path)
        xy = pd.read_csv(path)  # 使用pandas读取数据
        self.label = torch.from_numpy(xy.loc[:, 'IsAlert'].values).float()
        preprocessed_data = preprocessor.transform(xy.drop("IsAlert", axis=1))
        self.features = torch.from_numpy(preprocessed_data).float()
        self.len = xy.shape[0] - seq_num  # 可生成的数据（因为每10条实验数据生成一条时间序列的数据）

    def __getitem__(self, index):
        return self.features[index:index + self.seq_num], self.label[index + self.seq_num]

    def __len__(self):
        return self.len
