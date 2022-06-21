import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

# define neural net Q_\theta(s,a) as a class

class Qfunction(keras.Model):
    
    def __init__(self, nState, nAction, hidden_dims):
        """
        nState: dimension of state space
        nAction: dimension of action space
        hidden_dims: list containing output dimension of hidden layers 
        """
        super(Qfunction, self).__init__()

        # Layer weight initializer
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=(nState,))
        
        # Hidden Layer
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                      kernel_initializer=initializer)
            self.hidden_layers.append(layer) 
        self.output_layer = keras.layers.Dense(nAction, activation='linear', kernel_initializer=initializer) 
    
    @tf.function
    def call(self, states):
        x = self.input_layer(states)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)

# Wrapper class for training Qfunction and updating weights (target network) 

class DQN(object):
    
    def __init__(self, nState, nAction, hidden_dims, learning_rate):
        """
        nState: dimension of state space
        nAction: dimension of action space
        optimizer: 
        """
        self.qfunction = Qfunction(nState, nAction, hidden_dims)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.nState = nState
        self.nAction = nAction

    def _predict_q(self, states, actions):
        """
        states represent s_t
        actions represent a_t
        """
        #print(states, actions)
        q_ = self.compute_Qvalues(states)
        #print(q_)
        one_hot_action = tf.one_hot(actions, 2)
        #print(one_hot_action)
        predicts = tf.reduce_sum(one_hot_action * q_, axis=1)
        return predicts
        

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function
        """
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    
    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, nState], where numsamples is the number of samples
        output: Q values for these states. The output should have size 
        [numsamples, nAction] as numpy array
        """
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)


    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        with tf.GradientTape() as tape:
            Qpreds = self._predict_q(states, actions)
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        """
        We need a subroutine to update target network 
        i.e. to copy from principal network to target network. 
        This function is for copying  𝜃←𝜃target 
        """
        
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables
        
        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)


class ReplayBuffer(object):
    
    def __init__(self, maxlength, n_ensemble, bernoulli_prob):
        self.buffer = deque()
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob
        self.number = 0
        self.maxlength = maxlength
    
    def push(self, state, action, next_state, reward, done):
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble)
        self.buffer.append([state, action, next_state, reward, done, mask])
        self.number += 1
        if(self.number > self.maxlength):
            self.pop()
        
    def pop(self):
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]


class agent():
    
    def __init__(self, nState, nAction):
        self.nState = nState
        self.nAction = nAction
        self.hidden_dims = [10, 5]
        self.learning_rate = 0.3
        self.targetQ_list = [DQN(self.nState, self.nAction, self.hidden_dims, self.learning_rate) for _ in range(10)]
        self.principalQ_list = [DQN(self.nState, self.nAction, self.hidden_dims, self.learning_rate) for _ in range(10)]
        self.buffer = ReplayBuffer(10000, 10, 0.9)
        self.total_step = 0

    def action(self, state):
        vote_list = []
        for Qprincipal in self.principalQ_list:
            Q = Qprincipal.compute_Qvalues(state)
            action = np.argmax(Q)
            vote_list.append(action)
        return 1 if np.array(vote_list).sum() >= 5 else 0

    def load_weights(self):
        pass
        
    def observe(self, state, action, next_state, reward, done):
        self.buffer.push(state, action, next_state, reward, done)
    
    def update_after_step(self):
        if self.buffer.number < 20:
            return
        samples = self.buffer.sample(5)

        experiences = [[[],[],[]] for _ in range(10)]
        for sample in samples:
            mask = sample[-1]

            for index, item in enumerate(mask):
                if item == 1:
                    s = sample[0]
                    a = sample[1]
                    if sample[4]:
                        d = sample[3]
                    else:
                        d = sample[3] + 0.99 * np.max(self.targetQ_list[index].compute_Qvalues(sample[2]))
                    
                    experiences[index][0].append(s)
                    experiences[index][1].append(a)
                    experiences[index][2].append(d)

        for index, Qprincipal in enumerate(self.principalQ_list):
            Qprincipal.train(np.array(experiences[index][0]),np.array(experiences[index][1]), np.array(experiences[index][2]))

        self.total_step += 1

        if self.total_step % 100 == 0 :
            for i in range(10):
                self.targetQ_list[i].update_weights(self.principalQ_list[i])