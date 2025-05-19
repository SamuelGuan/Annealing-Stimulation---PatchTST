import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# 加载数据
class DataProcessor:
    def __init__(self, file_path, split_rate):
        self.file_path = file_path
        self.split_rate = split_rate
        self.data = self.load_dataset(self.file_path)

    def load_dataset(self, file_path:str):
        data = pd.read_csv(filepath_or_buffer=file_path, sep=',')
        # 处理缺失值
        data = data.ffill()
        # 处理异常值
        for column in data.select_dtypes(include=[np.number]).columns:
            if column != 'date':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data

    def dataset_visualization(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['OT'], label='OT')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Visualization of OT')
        plt.legend()
        plt.show()

    def register_data(self, sequence_len, output_len):
        train_data = self.data.drop(columns=['date']).iloc[:int(len(self.data)*self.split_rate)].to_numpy()
        eval_data = self.data.drop(columns=['date']).iloc[int(len(self.data)*self.split_rate):].to_numpy()
        # 把模型维度存储下来
        self.d_model = train_data.shape[-1]

        self.train_data_list = []
        self.eval_data_list = []

        for i in range(0, train_data.shape[-2]-sequence_len-output_len, sequence_len//2):
            x = torch.tensor(train_data[i: i+sequence_len], dtype=torch.float32)
            y = torch.tensor(train_data[i+sequence_len: i+sequence_len+output_len], dtype=torch.float32)
            self.train_data_list.append((x, y))

        for i in range(0, eval_data.shape[-2]-sequence_len-output_len, sequence_len//2):
            x = torch.tensor(eval_data[i: i + sequence_len], dtype=torch.float32)
            y = torch.tensor(eval_data[i + sequence_len: i + sequence_len + output_len], dtype=torch.float32)
            self.eval_data_list.append((x, y))


    def get_dataloader(self, mini_batch_size:int, mode:str)->list:
        if mode == 'train':
            # 随机抽样索引（无放回）,抽取mini_batch_size个样本
            sample_indices = torch.randperm(len(self.train_data_list))[:mini_batch_size].tolist()
            # 根据索引获取样本，返回一个iterable
            return [self.train_data_list[i] for i in sample_indices]
        elif mode == 'eval':
            # 随机抽样索引（无放回）,抽取mini_batch_size个样本
            sample_indices = torch.randperm(len(self.eval_data_list))[:mini_batch_size].tolist()
            # 根据索引获取样本，返回一个iterable
            return [self.eval_data_list[i] for i in sample_indices]
        else:
            raise ValueError('mode should be train or eval')


    def get_d_model(self)->int:
        return self.d_model



if __name__ == '__main__':
    sequence_len = 96
    output_len = 48
    split_rate = 0.8  # 80%训练，20%验证
    device = 'cuda'  # 使用GPU，如果没有GPU可以改为'cpu'

    # 初始化数据处理对象
    data_processor = DataProcessor('../dataset/ETTh1.csv', split_rate)

    # 可视化数据
    data_processor.dataset_visualization()

    # 注册数据
    data_processor.register_data(sequence_len, output_len)

    # 创建数据加载器
    train_dataloader = data_processor.get_dataloader(mode='train', mini_batch_size=32)

    count = 0
    for inputs, outputs in train_dataloader:
        print(f"输入数据形状: {inputs.shape}")
        print(f"输出数据形状: {outputs.shape}")
        count += 1
        if count == 2:
            break
