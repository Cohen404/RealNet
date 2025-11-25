import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class FeatureSelectionEnv:
    """
    Environment for feature selection using reinforcement learning
    """
    
    def __init__(self, 
                 total_channels: int, 
                 initial_selection: List[int] = None,
                 alpha: float = 0.7, 
                 beta: float = 0.3, 
                 gamma: float = 0.1):
        """
        Initialize the feature selection environment
        
        Args:
            total_channels: Total number of channels available
            initial_selection: Initial channel selection (used as starting point)
            alpha: Weight for pixel-level AUC improvement
            beta: Weight for image-level AUC improvement
            gamma: Weight for channel selection penalty
        """
        self.total_channels = total_channels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize state
        if initial_selection is not None:
            # Filter out indices that are out of bounds
            self.current_selection = [idx for idx in initial_selection if 0 <= idx < total_channels]
        else:
            # Default to selecting first half of channels
            self.current_selection = list(range(min(total_channels // 2, total_channels)))
        
        # State representation: binary vector indicating selected channels
        self.state = np.zeros(total_channels)
        # Only set valid indices to 1
        for idx in self.current_selection:
            if 0 <= idx < total_channels:
                self.state[idx] = 1
        
        # History for computing delta metrics
        self.prev_pixel_auc = 0.0
        self.prev_image_auc = 0.0
        
        # Track episode
        self.episode_step = 0
        self.max_episode_steps = total_channels  # Max steps per episode
        
    def reset(self, initial_selection: List[int] = None) -> np.ndarray:
        """
        Reset the environment
        
        Args:
            initial_selection: Initial channel selection (optional)
            
        Returns:
            Initial state
        """
        if initial_selection is not None:
            # Filter out indices that are out of bounds
            self.current_selection = [idx for idx in initial_selection if 0 <= idx < self.total_channels]
        else:
            # Default to selecting first half of channels
            self.current_selection = list(range(min(self.total_channels // 2, self.total_channels)))
        
        # Reset state
        self.state = np.zeros(self.total_channels)
        # Only set valid indices to 1
        for idx in self.current_selection:
            if 0 <= idx < self.total_channels:
                self.state[idx] = 1
        
        # Reset history
        self.prev_pixel_auc = 0.0
        self.prev_image_auc = 0.0
        
        # Reset episode
        self.episode_step = 0
        
        return self.state.copy()
    
    def step(self, action: int, pixel_auc: float, image_auc: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Action to take (channel index to toggle)
            pixel_auc: Current pixel-level AUC
            image_auc: Current image-level AUC
            
        Returns:
            next_state: Next state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Apply action (toggle channel selection)
        if action in self.current_selection:
            self.current_selection.remove(action)
            self.state[action] = 0
        else:
            self.current_selection.append(action)
            self.state[action] = 1
        
        # Calculate reward using the specified formula
        delta_pixel_auc = pixel_auc - self.prev_pixel_auc
        delta_image_auc = image_auc - self.prev_image_auc
        channel_penalty = len(self.current_selection) / self.total_channels
        
        reward = (self.alpha * delta_pixel_auc + 
                 self.beta * delta_image_auc - 
                 self.gamma * channel_penalty)
        
        # Update previous metrics
        self.prev_pixel_auc = pixel_auc
        self.prev_image_auc = image_auc
        
        # Increment episode step
        self.episode_step += 1
        
        # Check if episode is done
        done = self.episode_step >= self.max_episode_steps
        
        # Prepare info
        info = {
            'selected_channels': len(self.current_selection),
            'channel_indices': self.current_selection.copy(),
            'delta_pixel_auc': delta_pixel_auc,
            'delta_image_auc': delta_image_auc,
            'channel_penalty': channel_penalty
        }
        
        return self.state.copy(), reward, done, info
    
    def get_available_actions(self) -> List[int]:
        """
        Get list of available actions
        
        Returns:
            List of available channel indices
        """
        return list(range(self.total_channels))
    
    def get_current_selection(self) -> List[int]:
        """
        Get current channel selection
        
        Returns:
            List of selected channel indices
        """
        return self.current_selection.copy()
    
    def get_state_dim(self) -> int:
        """
        Get state dimension
        
        Returns:
            State dimension
        """
        return self.total_channels
    
    def get_action_dim(self) -> int:
        """
        Get action dimension
        
        Returns:
            Action dimension
        """
        return self.total_channels