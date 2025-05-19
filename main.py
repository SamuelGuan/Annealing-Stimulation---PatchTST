from util.datahanding import DataProcessor
from util.AnnealingSimulation import AnnealingSimulation
from util.BackwardOptimizer import MixOptimizer
from util.patchTST import PatchTST
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math as m

torch.autograd.set_detect_anomaly(True)
class NetWorkTrainAndEvaluate(object):
    def __init__(self, file_path, mini_batch_size, train_test_split_rate=0.7, device='cuda'):
        self.device = device
        self.train_test_split_rate = train_test_split_rate
        self.mini_batch_size = mini_batch_size
        self.data = DataProcessor(file_path=file_path, split_rate=self.train_test_split_rate)

    def train(self,
            sequence_len:int,
            output_len:int,
            patch_size:int,
            beta:float,
            drop_last:bool,
            learning_rate:float,
            bias:bool,
            )->float:
        '''
        :param <int> sequence_len: the row of one example matrix
        :param <int> patch_size: the size of patching one example matrix
        :param <float> beta: the fuss factor between layer1 attn score after softmax and layer2 attn score after softmax
        :param <bool> drop_last: whether to drop last example which didn't have the same shape as other example
        :param <float> learning_rate: learning rate for optimizer
        :param <bool> bias: whether to use bias
        :return: <float> a loss decent rate
        '''

        # 处理已经加载好的数据
        self.data.register_data(sequence_len=sequence_len, output_len=output_len)

        # 获取数据集特征数量
        d_model = self.data.get_d_model()

        # 加载模型到gpu
        model = PatchTST(sequence_len=sequence_len,
                    d_model=d_model,
                    patch_size=patch_size,
                    output_len=output_len,
                    beta=beta,
                    bias=bias,
                    drop_last=drop_last,
                    device=self.device).to(self.device)

        # 设置初始学习率
        lr = learning_rate

        # 加载优化器
        optimizer = MixOptimizer(params=model.parameters(),
                                 lr=lr,
                                 alpha = 0.9,
                                 )

        # 加载学习率更新器
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=30,
            eta_min=0.000001,
            last_epoch=2,
        )

        # 设置损失函数
        criterion = nn.MSELoss().to(self.device)

        # 设置学习轮次
        epochs = 126

        # 设置损失值列表，用于画图
        epoch_mean_train_loss_list = []
        epoch_mean_eval_loss_list = []

        for epoch in range(epochs):
            epoch_train_loss = 0
            epoch_train_count = 0
            # 梯度累积缓冲区
            for param in model.parameters():
                param.grad = torch.zeros_like(param.data).to(self.device)

            # 采用 mini-batch的方式
            # mini-batch的大小在轮数较大的时候增大而增大
            for inputs, outputs in self.data.get_dataloader(mode='train', mini_batch_size=int(self.mini_batch_size*(1+2*epoch/epochs))):
                train_pred = model(inputs.to(self.device))
                loss = criterion(train_pred, outputs.to(self.device))
                loss.backward()
                epoch_train_loss += loss.item()
                epoch_train_count += 1

            # 计算平均损失
            epoch_mean_train_loss_list.append(epoch_train_loss / epoch_train_count)

            # 梯度归一化
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= epoch_train_count

            optimizer.step(epoch_factor = self.epoch_factor_cal(epoch=epoch,epochs=epochs,min_factor=0.7))
            lr_scheduler.step()
            optimizer.zero_grad()

            # 每 5轮打印一次训练损失
            if epoch % 5 == 0:
                print(f'epoch:[{epoch+1}/{epochs}], train_loss:[{epoch_mean_train_loss_list[-1]}]')

            if epoch >= epochs*0.5:
                # 在轮次小于训练进度的3/4的时候没必要检验
                epoch_eval_loss = 0
                epoch_eval_count = 0
                model.eval()
                with torch.no_grad():
                    for inputs, outputs in self.data.get_dataloader(mode='eval', mini_batch_size=32):
                        eval_pred = model(inputs.to(self.device))
                        epoch_eval_loss += criterion(eval_pred, outputs.to(self.device)).item()
                        epoch_eval_count += 1
                epoch_mean_eval_loss_list.append(epoch_eval_loss / epoch_eval_count)
                if epoch % 5 == 0:
                    print(f'epoch:[{epoch + 1}/{epochs}], eval_loss:[{epoch_mean_eval_loss_list[-1]}]')
                model.train()

        return float(np.mean(epoch_mean_eval_loss_list[-10:]))

    def epoch_factor_cal(self,epoch,epochs,min_factor=0.7):
        if epoch >= epochs:
            raise ValueError('epoch must be less than epochs!')

        factor = 1/(1+m.exp( m.exp(3)*(epoch/epochs-0.5) ))
        if factor <= min_factor:
            factor = min_factor
        return factor



if __name__ == '__main__':
    annealing_simulator = AnnealingSimulation()
    network = NetWorkTrainAndEvaluate(file_path=r'dataset/ETTh1.csv',
                                      mini_batch_size=32,
                                      train_test_split_rate=0.7,
                                      device='cuda')
    annealing_simulator.searchBestParamsGroup(train_func=network.train)


