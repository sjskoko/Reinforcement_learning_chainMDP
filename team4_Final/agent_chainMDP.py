import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import logging
import math


class FeatureNet(nn.Module): #from state vector, extract feature with 1d-convolution network
    
    def __init__(self, n_states, num_channels=1):
        super(FeatureNet, self).__init__()
        self.num_channels = num_channels
        #nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv1d(self.num_channels, 4, 3, 1)
        self.conv2 = nn.LazyConv1d(1, 3, 1)

        def conv1d_size_out(size, kernel_size, stride):
            return (size - (kernel_size -1) - 1) // stride + 1
        
        final_size = conv1d_size_out(n_states, 3, 1)
        final_size = conv1d_size_out(final_size, 3, 1)
        self.reshape_size = final_size
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.reshape_size)

        return x

        

class HeadNet(nn.Module): # 2-layer network -> input: feature vector , output: 2-dimension vector(value for each action)

    def __init__(self, reshape_size, n_actions):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    
    def __init__(self, n_ensemble, n_states, n_actions, num_channels):
        super(EnsembleNet, self).__init__()
        self.feature_net = FeatureNet(num_channels = num_channels)
        reshape_size = self.core_net.reshape_size
        self.net_list = nn.ModuleList([HeadNet(reshape_size = reshape_size, n_actions = n_actions) for k in range(n_ensemble)])

    def _feature(self, x):
        return self.feature_net(x)
    
    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k = None):
        if k is not None:
            return self.net_list[k](self.feature_net(x))
        else:
            feature_cache = self._feature(x)
            net_heads = self._heads(feature_cache)
            return net_heads

#remove history Dataset
class memoryDataset(object):

    def __init__(self, maxlen, n_ensemble=1, bernoulli_prob = 0.9):
        self.memory = deque(maxlen=maxlen)
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob

        if n_ensemble==1:
            self.bernoulli_prob = 1

        self.subset = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'life', 'terminal', 'mask'))

    def push(self, state, action, next_state, reward, done, life, terminal):
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array([done])
        life = np.array([life])
        terminal = np.array([terminal])
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble)

        self.memory.append(self.subset(state, action, next_state, reward, done, life, terminal, mask))
        
    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = self.subset(*zip(*batch))

        state = torch.tensor(np.stack(batch.state), dtype = torch.float)
        action = torch.tensor(np.stack(batch.action), dtype = torch.long)
        reward = torch.tensor(np.stack(batch.reward), dtype = torch.float)
        next_state = torch.tensor(np.stack(batch.next_state), dtype = torch.float)
        done = torch.tensor(np.stack(batch.next_state), dtype=torch.float)
        life = torch.tensor(np.stack(batch.life), dtype = torch.float)
        terminal = torch.tensor(np.stack(batch.mask), dtype = torch.float)
        batch = self.subset(state, action, next_state, reward, done, life, terminal, mask)

        return batch

class Update(object):

    """
    Perform an update step between a DQN and a target DQN
    """

    def __init__(self, dqn, target_dqn):

        self.dqn = dqn
        self.target_dqn = target_dqn

class agent():

    def __init__(self, nState , nAction, history_size, network_builder, replay_memory, num_heads, **kwargs):

        self.nAction = nAction
        self.history_size = history_size

        self.num_heads = num_heads
        self.head_number = 0

        # Need to query the replay memory for training examples
        self.replay_memory = replay_memory

        #Discount factor & minibatch size
        self.discount_factor = kwargs.get('discount_factor', 0.99)
        self.minibatch_size = kwargs.get('minibatch_size', 32)


