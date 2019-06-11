import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent
from environment import Environment
from collections import namedtuple, deque
import random
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt


 
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)#8 -> 64
        self.fc2 = nn.Linear(hidden_dim, action_num)#64 -> 4

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)#dim=1
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)#action_num = 4   state_dim = 8
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 1000 # total training episodes (actually too large...)100000
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        ##
        # 
        self.loss_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        ##
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards= []
        self.saved_actions = []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        # state = Variable(torch.tensor(state))
        # action = self.model(state)

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(m.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_actions, returns):#PolicyNet.saved_log_probs
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        # TODO:
        # compute loss
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]#PolicyNet.saved_log_probs
        # policy_loss.backward()
        # self.optimizer.step()

    def train(self):
        ##
        myInfo_reward = []
        myInfo_epochs = [] 
        ##

        avg_reward = None # moving average of reward
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            # print("state:",state.shape)
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                # self.saved_actions.append(m.log_prob(action))
                self.rewards.append(reward)
            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            ## learning curve info
            myInfo_epochs.append(epoch)
            myInfo_reward.append(avg_reward)

            # update model
            # Replay memory
            # self.memory = ReplayBuffer(4, BUFFER_SIZE, BATCH_SIZE, seed)
            self.update()#,self.gamma

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                
            
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break

        plt.plot(myInfo_epochs, myInfo_reward, 'b.-')
        plt.title('Learning Curve')
        plt.ylabel('Avg Reward')
        plt.xlabel('Epochs')
        plt.show()
        plt.savefig("learning_curve_pg.jpg")
