from chain_mdp import ChainMDP
from agent_chainMDP import agent

# recieve 1 at rightmost state and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()

""" Your agent"""
sa_list = []
for i in range(env.n):
    for j in [0,1]:
        sa_list.append((i, j))
agent_params = {'gamma'            : 0.9,
                'kappa'            : 1.0,
                'mu0'              : 0.0,
                'lamda'            : 4.0,
                'alpha'            : 3.0,
                'beta'             : 3.0,
                'max_iter'         : 100,
                'sa_list'          : sa_list}
agent = agent(agent_params)

done = False
cum_reward = 0.0
# always move right left: 0, right: 1
# action = 1
for episode in range(100):
    s = env.reset()
    for tau in range(18):
        a = agent.take_action(s, 0)
        # Step environment
        s_, r, t = env.step(a)
        agent.observe([t, s, a, r, s_])
        agent.update_after_step(10, True)
        # Update current state
        s = s_

s = env.reset()

while True: 
    action = agent.take_action(s, 0)
    s_, r, t = env.step(action)
    print(s_, r, t)
    cum_reward += r
    s = s_
    if t == 18:
        break
print(f"total reward: {cum_reward}")
