import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent for feature selection
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, device='cuda'):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = Memory()
        
    def select_action(self, state, available_actions=None):
        """
        Select action based on current policy
        Args:
            state: current state
            available_actions: list of available action indices (optional)
        Returns:
            action: selected action
            action_logprob: log probability of the action
        """
        state = torch.FloatTensor(state).to(self.device)
        
        # Handle batch vs single state
        if len(state.shape) > 1:
            # Batch processing - take first element for now
            # In a more complete implementation, you might want to handle each state separately
            state = state[0:1]  # Keep batch dimension but with single element
            
        with torch.no_grad():
            action_probs = self.policy.actor(state)
            
            # If available_actions is provided, mask unavailable actions
            if available_actions is not None:
                mask = torch.zeros_like(action_probs)
                mask[available_actions] = 1
                action_probs = action_probs * mask
                action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        return action.item(), action_logprob.item()
    
    def update(self):
        """
        Update policy using PPO algorithm
        """
        # Convert list to tensor
        old_states = torch.FloatTensor(self.memory.states).to(self.device)
        old_actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device)
        old_rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = old_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, old_rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear_memory()
        
    def save(self, path):
        """Save model parameters"""
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        """Load model parameters"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


class ActorCritic(nn.Module):
    """
    Actor-Critic model for PPO
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self):
        raise NotImplementedError
        
    def evaluate(self, state, action):
        """
        Evaluate action and state value
        Args:
            state: current state
            action: action taken
        Returns:
            action_logprob: log probability of the action
            state_value: value of the state
            dist_entropy: entropy of the action distribution
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class Memory:
    """
    Memory buffer for PPO
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]