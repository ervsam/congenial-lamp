import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

        self.episodes_in_buffer = 0

    def insert(self, obs_fovs, partial_prio, global_reward, local_rewards, groups):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((obs_fovs, partial_prio, global_reward, local_rewards, groups))

    def sample(self, batch_size):
        ep_ids = np.random.choice(
            len(self.buffer), batch_size, replace=False)
        obs_fovs, partial_prio, global_reward, local_rewards, groups = zip(
            *[self.buffer[i] for i in ep_ids])
        return torch.stack(obs_fovs), partial_prio, torch.tensor(global_reward), local_rewards, groups

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize the prioritized experience replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences the buffer can hold.
            alpha (float): How much prioritization is used. 0 corresponds to uniform sampling.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def insert(self, obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals, td_error=None):
        """
        Add a new experience to the buffer.
        
        Args:
            experience (tuple): A single experience tuple (state, action, reward, next_state, done).
            td_error (float): Temporal-Difference error used to calculate priority.
        """
        experience = (obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals) 
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        # Update priorities
        self.priorities[self.pos] = max_priority if td_error is None else td_error ** self.alpha

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences with priorities.
        
        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): To what degree to use importance-sampling weights (0 - no corrections, 1 - full correction).
        
        Returns:
            tuple: Batch of experiences and corresponding importance-sampling weights.
        """
        if len(self.buffer) == 0:
            raise ValueError("The replay buffer is empty.")

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals = zip(*[self.buffer[i] for i in indices])
        return torch.stack(obs_fovs), partial_prio, torch.tensor(global_reward), local_rewards, groups, starts, goals, indices

        # Importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return experiences
    # , indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities of sampled experiences.
        
        Args:
            indices (list): Indices of experiences to update.
            td_errors (list): Corresponding TD errors.
        """
        print("PrioritizedReplayBuffer.update_priorities error: ", td_errors)
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)