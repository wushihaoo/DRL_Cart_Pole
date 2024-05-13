import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class DQN:
    def __init__(self, n_states, n_actions, n_hidden, memory_size, sample_size, learning_rate, discount_rate, target_update_interval):
        # 状态、动作、隐藏层特征维度
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        # 记忆库大小、采样大小、记忆库维度（状态特征维度*2+动作+奖励）
        self.memory_size = memory_size
        self.sample_size = sample_size
        self.memory = deque(maxlen=self.memory_size)
        # 优化器学习率，奖励折扣率
        self.lr = learning_rate
        self.gamma = discount_rate
        # 目标网络更新频率
        self.update_counter = 0
        self.target_update_interval = target_update_interval
        # 定义网络、优化器、损失函数
        self.eval_net = Net(self.n_states, self.n_actions, self.n_hidden)
        self.target_net = Net(self.n_states, self.n_actions, self.n_hidden)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_record = []

    # 选择动作
    def choose_action(self, state, epsilon):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        epsilon = epsilon
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net(state)
            action = actions_value.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    # 储存(s, a, r, s_)至记忆库    
    def store_transition(self, s, a, r, s_):
        self.memory.append(np.hstack((s, [a, r], s_)))

    # 更新网络
    def update(self):
        # 从记忆库中采样
        sample_memory = np.array(random.sample(self.memory, self.sample_size))
        sample_state = torch.FloatTensor(sample_memory[:, :self.n_states])
        sample_action = torch.LongTensor(sample_memory[:, self.n_states:self.n_states+1])
        sample_reward = torch.LongTensor(sample_memory[:, self.n_states+1:self.n_states+2])
        sample_next_state = torch.FloatTensor(sample_memory[:, -self.n_states:])

        # 更新eval_net
        q_eval = self.eval_net(sample_state).gather(1, sample_action)
        q_next_state = self.target_net(sample_next_state).max(1)[0].detach()
        q_target = sample_reward + self.gamma * q_next_state
        loss = self.loss(q_eval, q_target)
        self.loss_record.append(loss.detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新target_net
        if self.update_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.update_counter += 1

    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_record)), self.loss_record)
        plt.xlabel("agent_update_step")
        plt.ylabel("dqn_loss")
        plt.show()