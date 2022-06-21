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

        self.subset = namedtuple('Transition', ('state', 'action', 'reward', 'done',  'mask'))

    def push(self, state, action, next_state, reward, done):
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array([done])
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
        #life = torch.tensor(np.stack(batch.life), dtype = torch.float)
        terminal = torch.tensor(np.stack(batch.mask), dtype = torch.float)
        batch = self.subset(state, action, next_state, reward, done, mask)#, life, terminal)

        return batch


class DQNSolver():

    def __init__(self, nState, nAction, n_ensemble=10, env):

        self.env = env
        self.discount = 0.99
        self.nState = nState
        self.nAction = nAction
        self.action_space = np.range(self.nAction)
        self.max_steps = 18
        self.memory = memoryDataset(maxlen= 10, n_ensemble= n_ensemble, bernoulli_prob = bernoulli_prob)
        self.lr = 0.1
        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=self.lr)
        self.n_ensemble = n_ensemble
        self.policy_model = build_model()
        self.target_model = build_model()
    #choose action
    def choose_action(self, header_number:int=None, epsilon=None):
        if epsilon is not None:
            if np.random.random() <= epsilon:
                return self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to()
                    if header_numver is not None:
                        action = self.target_model(state, header_number).cpu()
                        return int(action.max(1).indices.numpy())
                    else:
                        actions = self.target_model(state)
                        actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                        actions = Counter(actions)
                        action = actions.most_common(1)[0][0]
                        return action
        else:
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to()
                if header_number is not None:
                    action = self.policy_model(state, header_number).cpu()
                    return int(action.max(1).indices.numpy())
                else:
                    actions = self.policy_model(state)
                    actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                    actions = Counter(actions)
                    action = actions.most_common(1)[0][0]
                    return action
    #def get_epsilon -> get epsilon increasing by time
    # update weight
    def replay(self, batch_size):

        self.optimizer.zero_grad()
        batch = self.memory.sample(batch_size)
        state = batch.state.to()
        action = batch.action.to()
        next_state = batch.next_state.to()
        reward = batch.reward
        reward = reward.type(torch.bool).type(torch.float).to(self.device)
        done = batch.done.to()
        mask = batch.mask.to()

        with torch.no_grad():
            next_state_action_values = self.policy_model(next_state)
        state_action_values = self.policy_model(state)

        total_loss = []
        for head_num in range(self.n_ensemble):
            total_used = torch.sum(maks[:, head_num])
            if total_used > 0.0:
                next_state_value = torch.max(next_state_action_values[head_num], dim = 1).values.view(-1, 1)
                reward = reward.view(-1, 1)
                target_state_value = torch.stack([reward + (self.discount * next_state_value), reward], dim = 1).squeeze().gather(1, terminal)
                state_action_value = state_action_values[head_num].gather(1, action)
                loss = F.smooth_l1_loss(state_action_value, target_state_value, reduction='none')
                loss = mask[:, head_num] * loss
                loss = torch.sum(loss/total_used)
                total_loss.append(loss)
        
        if len(total_loss) > 0:
            total_loss = sum(total_loss)/self.n_ensemble
            total_loss.backward()
            self.optimizer.step()

    def build_model(self):
          return EnsembleNet(self.n_ensemble, self.nState, self.nAction, 1) #1 = num_channels

    def train(self, num_episodes):
        tau = 100
        heads = list(range(self.n_ensemble))
        active_head = heads[0]


        for ite in range(num_episodes):
            
            s = self.env.reset()
            done = False
            rsum = 0

            while not done:
                step_count += 1
                action = self.choose_action(active_head, self.get_epsilon(step))
                next_state, reward, done _ = self.env.step(action)
                self.memory.push(state, action, next_state, reward, done)
                    
                if self.memory.__len__ > initialize:
                   self.replay(self.batch_size)

                if step_count % tau == 1:
                    self.target_model.load_state_dict(self.policy_model.state_dict()) #transfer weight


            np.random.shuffle(heads)
            active_head = heads[0]


        state = self.env.reset()
        done = False
        train_score = 0
        train_length = 0
        last_life = 0
        terminal = True

    def test(self):

        


class agent():

    def __init__(self, nState , nAction, network_builder, replay_memory, num_heads, **kwargs):

        self.nAction = nAction
        self.history_size = history_size

        self.num_heads = num_heads
        self.head_number = 0

        # Need to query the replay memory for training examples
        self.replay_memory = replay_memory

        #Discount factor & minibatch size
        self.discount_factor = kwargs.get('discount_factor', 0.99)
        self.minibatch_size = kwargs.get('minibatch_size', 32)
        if kwargs.get('mode') == 'train':
            DQNSolver.train()
        else:
            agent.test()
            
