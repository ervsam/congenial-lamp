import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def insert(self, obs_fovs, partial_prio, global_reward, local_rewards, groups):
        if len(self.buffer) == self.buffer_size:
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
    def __init__(self, capacity, alpha=0.8, freeze=False):
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

        self.freeze = freeze

    def insert(self, obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals, td_error=None):
        """
        Add a new experience to the buffer.
        
        Args:
            experience (tuple): A single experience tuple (state, action, reward, next_state, done).
            td_error (float): Temporal-Difference error used to calculate priority.
        """

        if len(self.buffer) >= 64 and self.freeze == True:
            return

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
        probabilities = (priorities + 1e-8) / (priorities + 1e-8).sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        # experiences = [self.buffer[idx] for idx in indices]

        obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals = zip(*[self.buffer[i] for i in indices])

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        global_reward = torch.tensor(global_reward, dtype=torch.float32)

        return obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals, indices, weights


    def update_priorities(self, indices, td_errors):
        """
        Update priorities of sampled experiences.
        
        Args:
            indices (list): Indices of experiences to update.
            td_errors (list): Corresponding TD errors.
        """
        # print("PrioritizedReplayBuffer.update_priorities error: ", td_errors)
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)
    
    def save_to_file(self, filename):
        """
        Print the last 500 experiences in the buffer to csv file.
        """
        with open(filename, 'w') as f:
            for experience in self.buffer[-500:]:
                # (obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals)
                obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals = experience

                # Convert each element to string and write to file
                # f.write(str(partial_prio) + ',')
                f.write(str(global_reward).replace(",", ";") + ',')
                f.write(str([round(x, 2) for x in local_rewards]).replace(",", ";") + ',')
                f.write(str(groups).replace(",", ";") + ',')
                f.write(str(starts).replace(",", ";") + ',')
                f.write(str(goals).replace(",", ";") + '\n')

        print(f"Buffer saved to {filename}")