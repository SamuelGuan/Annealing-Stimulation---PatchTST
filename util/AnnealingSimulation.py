import itertools
import time
import math
import numpy as np
import pandas as pd
import copy
from util.parameters import annealingSimulation_parameters

class AnnealingSimulation(object):

    __all_strategy__ = ['fix','loss']

    def __init__(self):
        self.cooling_strategy = annealingSimulation_parameters['cooling_strategy']
        if self.cooling_strategy not in self.__all_strategy__:
            raise ValueError("""
            Cooling strategy Error! Choose from 'fix', 'loss' and 'epoch'!
            """)
        if self.cooling_strategy == 'fix':
            self.fix_rate = float(eval(input('请输入降温的固定比率：')))
            self.fix= True
        else:
            self.fix = False
            self.best_validation_loss_dict = dict()

        self.start_temperature = annealingSimulation_parameters['start_temperature']
        self.end_temperature = annealingSimulation_parameters['end_temperature']
        self.optimized_params_dict = annealingSimulation_parameters['optimized_parameters']
        self.current_temperature = self.start_temperature

        self.logger = list()

    def searchBestParamsGroup(self, train_func):
        # 获取所有列表的值
        value_lists = list(self.optimized_params_dict.values())
        # 计算参数组合的笛卡尔积
        cartesian_product = itertools.product(*value_lists)
        # 将笛卡尔积结果转换为字典形式
        all_param_group = []
        for product in cartesian_product:
            new_dict = dict(zip(self.optimized_params_dict.keys(), product))
            all_param_group.append(new_dict)

        num_param_group = len(all_param_group)

        print('\033[32m'+'【搜索空间信息】'+'\033[0m')
        print(f'搜索的参数名称分别为{self.optimized_params_dict.keys()}')
        print('共有'+'\033[32m'+f'{num_param_group}'+'\033[0m'+'组可搜索参数')
        print(f'温度下降的策略为{self.cooling_strategy}')
        print('\033[32m'+'【开始搜索最佳参数组合】'+'\033[0m')

        # 用于记录第一个参数为固定量的时候模型在参数组的损失量
        validation_loss_list =[]
        current_params_group_start_value = all_param_group[0][list(self.optimized_params_dict.keys())[0]]
        count = 1

        search_start_time = time.time()
        for model_param_group in all_param_group:
            start_time = time.time()
            print(f'\n当前为第{count}组可搜索参数,剩余{num_param_group-count}组,进度{round(count/num_param_group,5)}%')
            # 获取第一个参数为固定量的值
            previous_params_group_start_value = current_params_group_start_value
            current_params_group_start_value = model_param_group[list(self.optimized_params_dict.keys())[0]]
            # -当本轮的第一个参数为固定量的值与上一轮不相等的时候，一定说明了模型参数进入了下一轮贪心搜索
            # 则此时需要清空上一个大范围的损失列表，给本轮的损失值腾出空间,并重置温度
            # -当本轮的第一个参数为固定量的值与上一轮相等的时候， 如果温度已到结束温度，则接下来的都不必搜索
            if current_params_group_start_value != previous_params_group_start_value:
                    self.current_temperature = self.start_temperature
                    validation_loss_list.clear()
            else:
                if self.current_temperature <= self.end_temperature:
                    count += 1
                    model_param_group.update(keyword='validation_loss', value=validation_loss_list[-1])
                    self.logger.append(model_param_group)
                    continue
            # <dict> model_param_group:{
            #           'sequence_len':int,
            #           'output_len':int,
            #           'patch_size':int,
            #           'beta':float,
            #           "keep_shape":bool,
            #           'learning_rate':float,
            #           'bias':bool,
            #           'validation_loss':list
            #           }
            # 由于模型（train_func）不需要 ‘validation_loss’,所以参数传递之前需要进行一次深拷贝筛选
            model_param_group_without_validation_loss = {k: copy.deepcopy(model_param_group[k])
                                                                  for k in self.optimized_params_dict.keys()
                                                                  if k in model_param_group}
            # 将模型参数丢给训练函数开始初始化模型与训练，返回的本轮验证集的损失存储在列表中
            validation_loss = train_func(**model_param_group_without_validation_loss)
            if not isinstance(validation_loss, float):
                raise ValueError("function's return " + "\033[32m" + "<train_func> " + "\033[0m" + f"must be float, but get {type(validation_loss)}!")
            print(f'=>本组参数为 {model_param_group}')
            print(f'=>本组参数的验证集损失为 [{validation_loss}],耗时 [{time.time()-start_time}]s')
            validation_loss_list.append(validation_loss)
            # 温度更新
            if self.fix:
                # 固定比率更新
                self.current_temperature *= self.fix_rate
            else:
                # 基于损失改进率的自适应温度调整策略
                if len(validation_loss_list) >= 2:
                    loss_improve_rate = 1 - (validation_loss_list[-1]/validation_loss_list[-2])
                    if loss_improve_rate > 0:
                        self.current_temperature *= math.exp(loss_improve_rate-1)\
                                if 1/loss_improve_rate <=self.start_temperature else self.start_temperature
                    else:
                        self.current_temperature *= math.exp(loss_improve_rate)
            #组数加一，并记录相关数据到日志
            count += 1
            model_param_group.update(keyword='validation_loss', value=validation_loss)
            self.logger.append(model_param_group)
        search_total_time = time.time() - search_start_time
        self.getSearchReport(key=self.optimized_params_dict.keys(),
                             num=num_param_group,
                             total_time=search_total_time,
                             strategy=self.cooling_strategy)

    def getSearchReport(self, key, num, total_time, strategy):
        # 统计损失值的分布，计算最大值，最小值，均值与标准差
        losses = np.array([lambda x:group['validation_loss'] for group in self.logger])
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        # 找出最好的一组参数，输出该参数组合配置与损失
        dataframe = pd.DataFrame(self.logger)
        best_params_group = dataframe.sort_values(by=['validation_loss'], ascending=False).iloc(0)

        print('\033[32m'+'/////////////////////////////////////////////////////////////////////'+'\033[0m')
        print('\033[32m'+'////////////////////////【参数组搜索报告】//////////////////////////////'+'\033[0m')
        print('\033[32m'+'/////////////////////////////////////////////////////////////////////'+'\033[0m')
        print('\n\n')
        print('\033[32m'+'[基础信息]'+'\033[0m')
        print('搜索参数为：',key)
        print('参数组搜索数量：',num)
        print('共计花费时间：',total_time,'s')
        print('参数组降温策略：',strategy)
        print('\033[32m'+'[统计信息]'+'\033[0m')
        print('模型验证集损失最大值：', max_loss)
        print('模型验证集损失最小值：', min_loss)
        print('模型验证集损失均值：', mean_loss)
        print('模型验证集损失方差：', std_loss)
        print('\033[32m'+'[最佳参数组]'+'\033[0m')
        print(best_params_group)
