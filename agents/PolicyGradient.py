import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    

class PolicyGradient:
    def __init__(self, n_states, n_actions, n_hidden, learning_rate, discount_rate):
        # 状态、动作、隐藏层特征维度
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        # 记忆库
        self.memory = []
        # 优化器学习率，奖励折扣率
        self.lr = learning_rate
        self.gamma = discount_rate
        # 定义网络、优化器
        self.policy_net = Net(self.n_states, self.n_actions, self.n_hidden)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_record = []

    # 选择动作
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        # print("action probs: ", probs)
        return action

    # 储存(s, a, r, s_)至记忆库    
    def store_transition(self, s, a, r, s_):
        self.memory.append(np.hstack((s, [a, r], s_)))

    # 更新网络
    def update(self):
        # 一个episode中的所有链信息
        sample_memory = np.array(self.memory)
        sample_state = torch.FloatTensor(sample_memory[:, :self.n_states])
        sample_action = torch.LongTensor(sample_memory[:, self.n_states:self.n_states+1])
        sample_reward = torch.LongTensor(sample_memory[:, self.n_states+1:self.n_states+2])

        # 更新plicy_net
        G = 0
        episode_loss = []
        self.optimizer.zero_grad()
        for i in reversed(range(len(sample_memory))):
            action_prob = self.policy_net(sample_state[i].unsqueeze(0)).gather(1, sample_action[i].unsqueeze(1))
            log_action_prob = torch.log(action_prob)
            G = sample_reward[i] + self.gamma*G
            loss = -log_action_prob * G
            episode_loss.append(loss.item())
            loss.backward()

        self.loss_record.append(np.mean(np.array(episode_loss)))        
        self.optimizer.step()


    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_record)), self.loss_record)
        plt.xlabel("agent_update_step")
        plt.ylabel("prolicy_net_loss")
        plt.show()