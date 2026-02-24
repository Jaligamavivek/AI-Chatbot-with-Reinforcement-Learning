"""
Q-Learning Reinforcement Learning Agent for the chatbot.
Learns optimal response selection based on user feedback.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

class QLearningAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state-action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Track state-action visits for exploration bonus
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        # Store last state-action for learning
        self.last_state = None
        self.last_action = None
        
        # Performance tracking
        self.total_rewards = 0
        self.episode_count = 0
        self.learning_history = []
    
    def get_state_action_key(self, state: str, action: str) -> str:
        """Create a unique key for state-action pair."""
        return f"{state}||{action}"
    
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """
        Select action using epsilon-greedy policy with exploration bonus.
        
        Args:
            state: Current state representation
            available_actions: List of possible actions (responses)
            
        Returns:
            Selected action (response)
        """
        if not available_actions:
            return "I'm not sure how to respond to that."
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose random action
            action = random.choice(available_actions)
        else:
            # Exploitation: choose action with highest Q-value + exploration bonus
            action_values = {}
            
            for action in available_actions:
                q_value = self.q_table[state][action]
                
                # Add exploration bonus for less visited state-action pairs
                visit_count = self.visit_counts[state][action]
                exploration_bonus = 1.0 / (1.0 + visit_count) if visit_count > 0 else 1.0
                
                action_values[action] = q_value + 0.1 * exploration_bonus
            
            # Select action with highest value
            action = max(action_values, key=action_values.get)
        
        # Store for learning
        self.last_state = state
        self.last_action = action
        
        # Update visit count
        self.visit_counts[state][action] += 1
        
        return action
    
    def update_q_value(self, reward: float, next_state: Optional[str] = None, 
                      next_actions: Optional[List[str]] = None):
        """
        Update Q-value based on received reward.
        
        Args:
            reward: Reward received for the last action
            next_state: Next state (optional, for multi-step learning)
            next_actions: Available actions in next state (optional)
        """
        if self.last_state is None or self.last_action is None:
            return
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        # Calculate next state value (for multi-step learning)
        next_q_max = 0.0
        if next_state and next_actions:
            next_q_values = [self.q_table[next_state][action] for action in next_actions]
            next_q_max = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q_max - current_q
        )
        
        self.q_table[self.last_state][self.last_action] = new_q
        
        # Update performance tracking
        self.total_rewards += reward
        self.episode_count += 1
        
        # Store learning history
        self.learning_history.append({
            'episode': self.episode_count,
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'q_value': new_q,
            'epsilon': self.epsilon
        })
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_reward_from_feedback(self, feedback_type: str, feedback_value: float) -> float:
        """
        Convert user feedback to reward signal.
        
        Args:
            feedback_type: Type of feedback ('like', 'dislike', 'rating', 'continue')
            feedback_value: Numerical value of feedback
            
        Returns:
            Reward value for RL learning
        """
        reward_mapping = {
            'like': 1.0,
            'dislike': -1.0,
            'rating': (feedback_value - 3.0) / 2.0,  # Scale 1-5 rating to -1 to 1
            'continue': 0.1,  # Small positive reward for conversation continuation
            'end': -0.1,  # Small negative reward for conversation ending
            'correction': -0.5,  # Negative reward for user corrections
            'repeat': -0.3  # Negative reward for repetitive responses
        }
        
        return reward_mapping.get(feedback_type, 0.0)
    
    def get_q_table_stats(self) -> Dict:
        """Get statistics about the Q-table for analysis."""
        if not self.q_table:
            return {'states': 0, 'actions': 0, 'total_entries': 0}
        
        total_entries = sum(len(actions) for actions in self.q_table.values())
        unique_states = len(self.q_table)
        unique_actions = len(set(
            action for actions in self.q_table.values() for action in actions.keys()
        ))
        
        # Calculate Q-value statistics
        all_q_values = [
            q_val for actions in self.q_table.values() 
            for q_val in actions.values()
        ]
        
        stats = {
            'states': unique_states,
            'actions': unique_actions,
            'total_entries': total_entries,
            'avg_reward': self.total_rewards / max(1, self.episode_count),
            'episodes': self.episode_count,
            'epsilon': self.epsilon
        }
        
        if all_q_values:
            stats.update({
                'avg_q_value': np.mean(all_q_values),
                'max_q_value': np.max(all_q_values),
                'min_q_value': np.min(all_q_values)
            })
        
        return stats
    
    def save_q_table(self, database_manager):
        """Save Q-table to database."""
        for state, actions in self.q_table.items():
            for action, q_value in actions.items():
                key = self.get_state_action_key(state, action)
                database_manager.save_q_value(key, q_value)
    
    def load_q_table(self, database_manager):
        """Load Q-table from database."""
        q_data = database_manager.load_q_table()
        
        for key, q_value in q_data.items():
            if '||' in key:
                state, action = key.split('||', 1)
                self.q_table[state][action] = q_value
    
    def get_best_actions_for_state(self, state: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k best actions for a given state."""
        if state not in self.q_table:
            return []
        
        actions = self.q_table[state]
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_actions[:top_k]
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.last_state = None
        self.last_action = None