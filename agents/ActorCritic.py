import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt



class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class CriticNet(nn.Module):
    def __init__(self, n_states, n_hidden):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    


    

class ActorCritic:
    def __init__(self, n_states, n_actions, n_hidden, actor_learning_rate, critic_learning_rate, discount_rate):
        # 状态、动作、隐藏层特征维度
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        # 记忆库
        self.memory = []
        # 优化器学习率，奖励折扣率
        self.ac_lr = actor_learning_rate
        self.cr_lr = critic_learning_rate
        self.gamma = discount_rate
        # 定义网络、优化器
        self.actor_net = ActorNet(self.n_states, self.n_actions, self.n_hidden)
        self.critic_net = CriticNet(self.n_states, self.n_hidden)        
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.ac_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.cr_lr)
        self.actor_loss_record = []
        self.critic_loss_record = []


    # 选择动作
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    # 储存(s, a, r, s_)至记忆库    
    def store_transition(self, s, a, r, d, s_):
        self.memory.append(np.hstack((s, [a, r, d], s_)))

    # 更新网络
    def update(self):
        # 一个episode中的所有链信息
        sample_memory = np.array(self.memory)
        sample_state = torch.FloatTensor(sample_memory[:, :self.n_states])
        sample_action = torch.LongTensor(sample_memory[:, self.n_states:self.n_states+1])
        sample_reward = torch.LongTensor(sample_memory[:, self.n_states+1:self.n_states+2])
        sample_done = torch.LongTensor(sample_memory[:, self.n_states+2:self.n_states+3])
        sample_next_state = torch.FloatTensor(sample_memory[:, -self.n_states:])

        v_sample_state = self.critic_net(sample_state)
        v_sample_next_state = self.critic_net(sample_next_state)
        td_target = sample_reward + self.gamma*v_sample_next_state*(1-sample_done)
        td_erro = td_target - v_sample_state
        log_prob = torch.log(self.actor_net(sample_state).gather(1, sample_action))
        critic_loss = torch.mean(F.mse_loss(self.critic_net(sample_state), td_target.detach()))
        actor_loss = torch.mean(-log_prob * td_erro.detach())

        self.actor_loss_record.append(actor_loss.item())
        self.critic_loss_record.append(critic_loss.item())
       
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

                

    def plot_loss(self):
        plt.plot(self.actor_loss_record)
        plt.xlabel("agent_update_step")
        plt.ylabel("actor_loss")
        plt.show()
        plt.plot(self.critic_loss_record)
        plt.xlabel("agent_update_step")
        plt.ylabel("critic_loss")
        plt.show()