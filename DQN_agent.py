"""
DQN agent
"""

import numpy as np
import torch
from torch import optim
from torch.nn import RNN, LSTM, LSTMCell
import torch.nn as nn
import torch.nn.functional as F
import random,math

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=5, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(5 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        x = input
        x, (h_n, c_n) = self.lstm(x)

        output_f = h_n[-2, :, :]
        output_b = h_n[-1, :, :]
        output = torch.cat([output_f, output_b], dim=-1)
        out_fc1 = self.fc1(output)
        out_relu = F.relu(out_fc1)
        out = self.fc2(out_relu)
        # 概率
        return F.log_softmax(out, dim=-1)

class DQN():
    def __init__(
            self,
            learning_rate=0.01,
            reward_decay=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy=0.9,
            e_greedy_increment=None,
            output_graph=False,
    ):
        #传递参数
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        #总学习步数
        self.learn_step = 0

        #初始化经验池
        #做成array方便运算
        #做成五个array
        #定大小和格式
        self.memory_s = np.zeros((self.memory_size,250,50))
        self.memory_a = np.zeros(self.memory_size)
        self.memory_r = np.zeros(self.memory_size)
        self.memory_s_ = np.zeros((self.memory_size,250,50))
        self.memory_d = np.zeros(self.memory_size)

        #建立网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_eval = RNN().to(self.device)
        self.net_target = RNN().to(self.device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=self.lr)
        self.critetrion = nn.CrossEntropyLoss().to(self.device)


    def choose_action(self,observation):
        if np.random.uniform() < self.epsilon:
            x = observation
            x = torch.unsqueeze(torch.FloatTensor(x), 0)
            input_ = torch.tensor(x, dtype=torch.float32).to(self.device)
            output = self.net_eval(input_)
            action = torch.max(output,1)[1][0].numpy() ##############???????????输出应该是【0】或者【1】简化为0，1？
            print(action)
        else:
            action = np.random.randint(0,2)
        return action


    def store_transition(self,s,a,r,s_,done):
        #读数据分别存在四个array中
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        self.memory_s[index] = s
        self.memory_a[index] = a
        self.memory_r[index] = r
        self.memory_s_[index] = s_
        self.memory_d[index] = done
        self.memory_counter += 1


    def learn(self):
        #更新target网络
        if self.learn_step % self.replace_target_iter == 0:
            self.net_target.load_state_dict(self.net_eval.state_dict())#######?????要不要to device?
        
        #经验池抓取
        sample_index = np.random.choice(self.memory_size,self.batch_size)
        #生成index直接在四个array里取，获得batch
        s_batch = np.array([self.memory_s[i] for i in sample_index])
        a_batch = np.array([self.memory_a[i] for i in sample_index])
        r_batch = np.array([self.memory_r[i] for i in sample_index])
        s__batch = np.array([self.memory_s_[i] for i in sample_index])
        d_batch = np.array([self.memory_d[i] for i in sample_index])
        #输入网络
        input_ = torch.tensor(s_batch, dtype=torch.float32).to(self.device)
        eval = self.net_eval(input_)
        next_input_ = torch.tensor(s__batch, dtype=torch.float32).to(self.device)
        target = self.net_target(next_input_)
        action = torch.tensor(a_batch, dtype=torch.float32).to(self.device)
        reward = torch.tensor(r_batch, dtype=torch.float32).to(self.device)
        done = torch.tensor(d_batch, dtype=torch.float32).to(self.device)
        #构建目标函数
        eval = eval.gather(1, action.unsqueeze(1).type(torch.int64)).squeeze(1)
        target = target.max(1)[0]
        target = reward + self.gamma * target * (1 - done)
        loss = (eval - target).pow(2).mean()##这里开始不确定怎么写，不清楚怎么把数据传到硬件上，以及哪些要传哪些不要传
        #更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        






    