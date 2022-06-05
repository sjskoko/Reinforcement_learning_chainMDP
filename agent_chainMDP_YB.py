import numpy as np
from chain_mdp import ChainMDP

class agent():

    def __init__(self):
        self._env = ChainMDP(10)
        self._n = self._env.n
        self._state = self._env.state
        #self._nsteps = self._env.nsteps
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._max_nsteps = self._env.max_nsteps
        self.num_episodes = 1000
        self.pi = self.train()

    def action(self, state):

        print(self.pi)
        return self.pi[state]

    
    def train(self):
        Q = np.zeros([self._observation_space.n, self._action_space.n])
        pi = np.random.randint(low=self._action_space.n, size=self._observation_space.n)
        count = np.ones([self._observation_space.n, self._action_space.n])/1000 # initialize to 0.001
        alpha = 0.8
        total_timestep = 0
        n = 10 #n-step SARSA
        epsilon = 0.1
        for i in range(self.num_episodes):
            #print(Q)
            #print(count)
            s = self._env.reset()
            d = False
            #a = pi[self._env.state]
            states = np.int_(np.zeros([self._max_nsteps]))
            actions = np.int_(np.zeros([self._max_nsteps]))
            rewards = np.zeros([self._max_nsteps+1])
            states[0] = self._env.state
            #actions[0] = a
            if np.random.rand(1) < epsilon or np.max(Q[s,:]) == 0:
                a = self._action_space.sample()
            else:
                a = pi[s]
            actions[0] = a
            #print(actions[0])
            for t in range(self._max_nsteps):
            
                s1, r, d, _ = self._env.step(int(actions[t]))
                total_timestep += 1
                self._env.nsteps += 1
                count[int(states[t]), int(actions[t])] += 1
                if np.random.rand(1) < epsilon or np.max(Q[s, :]) == 0:
                    a1 = self._action_space.sample()
                else:
                    a1 = pi[s1]
                rewards[t+1] = r

                if t < self._max_nsteps - 1:
                    actions[t+1] = a1
                    states[t+1] = s1
                
                tau = self._env.nsteps - n + 1

                if tau >= 0:
                    G = 0
                    for j in range(tau+1, min(tau+n, self._max_nsteps)+1):
                        G += rewards[j]
                    if tau + n < self._max_nsteps:
                        #print(states[tau])
                        G += Q[states[tau+n], actions[tau+n]]
                        Q[states[tau], actions[tau]] = Q[states[tau], actions[tau]] + alpha*(G - Q[states[tau+n], actions[tau+n]])
                    for j in range(self._observation_space.n):
                        pi[j] = np.argmax(Q[j, :] + np.sqrt(2*np.log(total_timestep)/count[j, :]))
                #print(count)
            for j in range(self._observation_space.n):
                pi[j] = np.argmax(Q[j, :] + np.sqrt(2*np.log(total_timestep)/count[j, :]))  #UCB count update
            #print(rewards)
            #print(Q)
            #print(count)
            #print(pi)
        return pi



class agent():
    
    def __init__(self):
        
        return
    
    def action(self):
        
        return 0
