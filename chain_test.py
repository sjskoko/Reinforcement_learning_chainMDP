from chain_mdp import ChainMDP
from agent_chainMDP_new import agent

# recieve 1 at rightmost state and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()

agent = agent(10, 2)

def training(k):
   for episode in range(k):
       s = env.reset()
       done = False

       cum_reward = 0
       while not done:
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            agent.observe(s, action, ns, reward, done)
            agent.update_after_step()

            #####################
            s = ns
       print(episode, cum_reward)
#training for 1000 episodes
training(10)


cum_reward = 0.0
s = env.reset()
done = False

while not done: 
    action = agent.action(s)
    ns, reward, done, _ = env.step(action)
    cum_reward += reward
    print(s, action, reward)
    cum_reward += reward
    s = ns

print(f"total reward: {cum_reward}")
