import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from RL.models.MLP import DiscreteMLP
from RL.utils.memory import ReplayBuffer
from RL.environments.cartpole import CartPole as env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

class DQNAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, batch_size=64, gamma=0.99, lr=1e-4):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DiscreteMLP(state_dim, hidden_dim, action_dim).to(self.device)
        # Target Q Network
        self.target_q_net = DiscreteMLP(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        
        # Replay Buffer - larger capacity for better stability
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size
        
        # Epsilon Greedy
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Target Network Update
        self.target_update_freq = 10
    
    def act(self, state, eval=False):
        if not eval and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.q_net(state)
            action = torch.argmax(scores)
            
        return action.item()
    
    def learn(self, state, action, reward, next_state, done):
                                
        self.replay_buffer.add((state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        state, action, reward, next_state, done = zip(*batch)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        
        y_hat = self.q_net(state).gather(1, action.unsqueeze(1))
        
        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_q_net(next_state).gather(1, next_actions)
            y = reward.unsqueeze(1) + self.gamma * next_q_values * (1 - done.unsqueeze(1))
        
        self.optimizer.zero_grad()  
        loss = self.loss_fn(y, y_hat)
        loss.backward()
        self.optimizer.step()

dqn_agent = DQNAgent(state_dim, hidden_dim, action_dim)

episode_scores = []
for episode in range(10000):
    state, info = env.reset()
    score = 0
    
    while True:
        action = dqn_agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        dqn_agent.learn(state, action, reward, next_state, done)
        state = next_state
        score += reward
        
        if done:
            break
        
    # Epsilon decay every episode
    dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon * dqn_agent.epsilon_decay)
    episode_scores.append(score)
    
    # Target Network Update
    if episode % dqn_agent.target_update_freq == 0:
        dqn_agent.target_q_net.load_state_dict(dqn_agent.q_net.state_dict())
        avg_score = np.mean(episode_scores)
        print(f"Episode {episode}, Avg Score {avg_score:.2f}, Epsilon {dqn_agent.epsilon:.3f}")
        
        # Early stopping if solved
        if avg_score >= 475:
            print(f"🎉 Environment solved in {episode} episodes!")
            break
            
        episode_scores = []

### Save Model ###
torch.save(dqn_agent.q_net.state_dict(), "ddqn_cartpole.pth")

### Evaluation ###
import gymnasium as gym
agent = DiscreteMLP(state_dim, hidden_dim, action_dim)
agent.load_state_dict(torch.load("ddqn_cartpole.pth"))

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

while True:
    action = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    
    if done:
        break