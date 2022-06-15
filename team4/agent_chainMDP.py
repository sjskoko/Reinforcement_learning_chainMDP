import numpy as np
from numpy.random import normal, gamma
from copy import deepcopy

def solve_tabular_continuing_PI(P, R, gamma, max_iter):
    '''
        Solves the Bellman equation for a continuing tabular problem.

        Returns greedy policy pi and corresponding Q-values.
    '''
    
    num_s, num_a = P.shape[:2]
    s_idx = np.arange(num_s)
    
    ones = np.eye(num_s)
    pi = np.zeros(num_s, dtype=np.int32)
    Q = None
    
    P_R = np.einsum('ijk, ijk -> ij', P, R)
    
    for i in range(max_iter):
    
        # Solve for Q values
        V = np.linalg.solve(ones - gamma * P[s_idx, pi, :], P_R[s_idx, pi])
        Q = P_R + gamma * np.einsum('ijk, k -> ij', P, V)

        # Get greedy policy - break ties at random
        pi_ = np.array([np.random.choice(np.argwhere(Qs == np.amax(Qs))[0]) \
                        for Qs in Q])
        
        if np.prod(pi_ == pi) == 1:
            break
        else:
            pi = pi_

    return pi, Q

def normal_gamma(mu0, lamda, alpha, beta):
    """
        Returns samples from Normal-Gamma with the specified parameters.
        
        Number of samples returned is the length of mu0, lambda, alpha, beta.
    """    

    # Check if parameters are scalars or vetors
    if type(mu0) == float:
        size = (1,)
    else:
        size = mu0.shape
        
    # Draw samples from gamma (numpy "scale" is reciprocal of beta)
    taus = gamma(shape=alpha, scale=beta**-1, size=size)

    # Draw samples from normal condtioned on the sampled precision
    mus = normal(loc=mu0, scale=(lamda * taus)**-0.5, size=size)
    
    return mus, taus

class agent():
    
    def __init__(self):
        sa_list = []

        for i in range(10):
            for j in range(2):
                sa_list.append((i, j))

        agent_params = {'gamma'            : 0.9,
                        'kappa'            : 1.0,
                        'mu0'              : 0.0,
                        'lamda'            : 4.0,
                        'alpha'            : 3.0,
                        'beta'             : 3.0,
                        'max_iter'         : 100,
                        'sa_list'          : sa_list}

        # PSRL agent parameters
        self.gamma = agent_params['gamma']
        self.kappa = agent_params['kappa']
        self.mu0 = agent_params['mu0']
        self.lamda = agent_params['lamda']
        self.alpha = agent_params['alpha']
        self.beta = agent_params['beta']
        self.sa_list = agent_params['sa_list'] 
        self.max_iter = agent_params['max_iter'] 

        self.Ppost = {} #P posterior
        self.Rpost = {} #R posterior
        self.buffer = [] #buffer
        self.num_s = len(set([s for (s, a) in self.sa_list])) #state 수
        self.num_a = len(set([a for (s, a) in self.sa_list])) #num actions

        # Lists for storing P and R posteriors
        self.Ppost_log = [] #list for storing P posteriors
        self.Rpost_log = [] #list for storing R posterior

        # Dynamics posterior
        self.Ppost = self.kappa * np.ones((self.num_s, self.num_a, self.num_s)) #

        # Rewards posterior parameters for non-allowed actions
        Rparam = [-1e12, 1e9, 1e12, 1e9]
        Rparam = [[[Rparam] * self.num_s] * self.num_a] * self.num_s
        self.Rpost = np.array(Rparam)

        # Rewards posterior parameters for allowed actions
        Rparam = [self.mu0, self.lamda, self.alpha, self.beta]
        Rparam = np.array([Rparam] * self.num_s)
        for (s, a) in self.sa_list:
            self.Rpost[s, a, ...] = Rparam
                
        self.sample_posterior_and_update_continuing_policy()

    def add_observations(self, s, a, r, s_):
        """ Add observations to log. """

        s, a, r, s_ = [np.array([data]) for data in [s, a, r, s_]]
        
        if hasattr(self, 'train_s'):
            self.train_s = np.concatenate([self.train_s, s], axis=0)
            self.train_a = np.concatenate([self.train_a, a], axis=0)
            self.train_s_ = np.concatenate([self.train_s_, s_], axis=0)
            self.train_r = np.concatenate([self.train_r, r], axis=0)
        
        else:
            self.train_s = s
            self.train_a = a
            self.train_s_ = s_
            self.train_r = r

    def sample_posterior(self):

        # Initialise posterior arrays (dynamics 0, reward large negative)
        P = np.zeros((self.num_s, self.num_a, self.num_s))
        R = np.zeros((self.num_s, self.num_a, self.num_s))

        for s in range(self.num_s):
            for a in range(self.num_a):
                P[s, a, :] = np.random.dirichlet(self.Ppost[s, a])
        
        for s in range(self.num_s):
            for a in range(self.num_a):
                for s_ in range(self.num_s):
                    mu0, lamda, alpha, beta = self.Rpost[s, a, s_]
                    R[s, a, s_] = normal_gamma(mu0, lamda, alpha, beta)[0]

        return P, R

    def update_posterior(self):

        # Transition counts and reward sums
        p_counts = np.zeros((self.num_s, self.num_a, self.num_s))
        r_sums = np.zeros((self.num_s, self.num_a, self.num_s))
        r_counts = np.zeros((self.num_s, self.num_a, self.num_s))

        for (s, a, r, s_) in self.buffer:
            p_counts[s, a, s_] += 1
            r_sums[s, a, s_] += r
            r_counts[s, a, s_] += 1

        # Update dynamics posterior
        for s in range(self.num_s):
            for a in range(self.num_a):
                # Dirichlet posterior params are prior params plus counts
                self.Ppost[s, a] = self.Ppost[s, a] + p_counts[s, a]

        # Update rewards posterior
        for s in range(self.num_s):
            for a in range(self.num_a):
                for s_ in range(self.num_s):
                    
                    mu0, lamda, alpha, beta = self.Rpost[s, a, s_]
                    
                    # Calculate moments
                    M1 = r_sums[s, a, s_] / max(1, r_counts[s, a, s_])
                    M2 = r_sums[s, a, s_]**2 / max(1, r_counts[s, a, s_])
                    n = r_counts[s, a, s_]
                    
                    # Update parameters
                    mu0_ = (lamda * mu0 + n * M1) / (lamda + n)
                    lamda_ = lamda + n
                    alpha_ = alpha + 0.5 * n
                    beta_ = beta + 0.5 * n * (M2 - M1**2)
                    beta_ = beta_ + n * lamda * (M1 - mu0)**2 / (2 * (lamda + n))    

                    self.Rpost[s, a, s_] = np.array([mu0_, lamda_, alpha_, beta_])

        # Reset episode buffer
        self.buffer = []

    def observe(self, transition):
        t, s, a, r, s_ = transition
        self.add_observations(s, a, r, s_)
        self.buffer.append([s, a, r, s_])

    def update_after_step(self, max_buffer_length, log):
        # Log posterior values
        if log:
            self.Ppost_log.append(deepcopy(self.Ppost))
            self.Rpost_log.append(deepcopy(self.Rpost))

        if len(self.buffer) >= max_buffer_length:
            self.update_posterior()
            self.sample_posterior_and_update_continuing_policy()

    def sample_posterior_and_update_continuing_policy(self):
    
        # Sample dynamics and rewards posterior
        P, R = self.sample_posterior()
        
        # Solve Bellman equation by policy iteration
        pi, Q = solve_tabular_continuing_PI(P, R, self.gamma, self.max_iter)
        
        self.pi = pi

    def load_weights(self):

        self.Ppost = np.load('./ppost.npy')
        self.Rpost = np.load('./rpost.npy')

        P, R = self.sample_posterior()
        
        # Solve Bellman equation by policy iteration
        pi, Q = solve_tabular_continuing_PI(P, R, self.gamma, self.max_iter)
        
        self.pi = pi

    def action(self, state):\
        return self.pi[int(state.sum()-1)]