import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from RL.models.MLP import MLPDiscretePolicy, DiscreteMLP
from RL.utils.memory import ReplayBuffer
from RL.environments.cartpole import CartPole as env

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

### wandb
import wandb
run = wandb.init(
    entity="comibear",
    # Set the wandb project where this run will be logged.
    project="REINFORCE-cartpole",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "REINFORCE",
        "gamma": 0.99,
        "hidden_dim": 128,
    },
)
###

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

class ReinforceAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma=0.99, lr=1e-4):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MLPDiscretePolicy(state_dim, hidden_dim, action_dim).to(self.device)
        self.baseline_net = DiscreteMLP(state_dim, hidden_dim, 1).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_baseline = optim.Adam(self.baseline_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.episode_buffer = list()
        
    @torch.no_grad()
    def act(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return torch.multinomial(self.policy_net(state), 1) if train else self.policy_net(state).argmax(dim=1)
    
    def learn(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        if done:
            state, action, reward, next_state, done = zip(*self.episode_buffer)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)
            
            # reward to return
            returns = []
            G = 0
            for r in reversed(reward):
                G = r + self.gamma * G
                returns.append(G)
            returns = returns[::-1]
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            
            # baseline
            values = self.baseline_net(state).squeeze()
            value_loss = self.loss_fn(values, returns)
            
            self.optimizer_baseline.zero_grad()
            value_loss.backward()
            self.optimizer_baseline.step()
            
            # log probability
            advantages = returns - values.detach()
            log_prob = torch.log(self.policy_net(state).gather(1, action.unsqueeze(1))).squeeze()
            
            # loss
            policy_loss = -torch.mean(log_prob * advantages)
            
            # update policy network
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            run.log({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "length": len(self.episode_buffer),
                "avg_advantage": advantages.mean().item(),
            })
            
            self.episode_buffer.clear()
            
        return None
            
agent = ReinforceAgent(state_dim, hidden_dim, action_dim)

for episode in range(10000):
    state, info = env.reset()
    step_count = 0
    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        step_count += 1
        
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            break
        
    print(f"Episode {episode}, Length {step_count}")
    # if step_count >= 490:
    #     print(f"🎉 Environment solved in {episode} episodes!")
    #     import os
    #     torch.save(agent.policy_net.state_dict(), os.path.join(current_dir, "reinforce_cartpole.pth"))
    #     break

run.finish()

### Evaluation ###
import gymnasium as gym
agent = ReinforceAgent(state_dim, hidden_dim, action_dim)
agent.policy_net.load_state_dict(torch.load(os.path.join(current_dir, "reinforce_cartpole.pth")))

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

while True:
    action = agent.act(state, train=False)
    next_state, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    state = next_state
    if done:
        break