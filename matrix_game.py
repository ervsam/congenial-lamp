# %%
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
logger = Logger("mtx_game.txt")

# ENV PARAMETERS
TRAINING_STEPS = 20000
LR = 0.0005
LAMBDA = 1
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 20000

N_ACTIONS = 21

is_QTRAN_alt = True

# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def append(self, item):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size):
        # Randomly sample from replay_buffer.buffer
        batch = np.random.choice(
            len(self.buffer), size=batch_size, replace=False)
        sampled_batch = [self.buffer[i] for i in batch]

        return sampled_batch


# QTRAN MODEL
class Qtran(nn.Module):
    def __init__(self):
        super(Qtran, self).__init__()

        self.hidden_dim = 64

        self.enc = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.q_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, N_ACTIONS)
        )

        self.v_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )


        self.lin2 = nn.Sequential(
            nn.Linear(self.hidden_dim+N_ACTIONS, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.lin3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin4 = nn.Linear(self.hidden_dim, N_ACTIONS)

        # joint q net with 2 linear layers
        self.q_prime_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.ELU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x, b_act1, b_act2):
        batch_size = x.shape[0]
        h1 = self.enc(x[:, 0, :])
        h2 = self.enc(x[:, 1, :])

        q1, q2 = self.q_net(h1), self.q_net(h2)

        key2 = torch.stack([h1, h2], dim=1)
        assert key2.shape == (batch_size, 2, self.hidden_dim), f"Expected {(batch_size, 2, self.hidden_dim)}, got {key2.shape}"

        vtot = self.v_net(key2.sum(dim=1))
        assert vtot.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {vtot.shape}"

        a1_max = torch.argmax(q1, dim=1)
        a2_max = torch.argmax(q2, dim=1)

        joint_qs = []
        joint_qs_prereduced = []
        for a1, a2, e1, e2 in zip(b_act1, b_act2, h1, h2):
            # one hot encode actions
            a1 = F.one_hot(a1, num_classes=N_ACTIONS)
            a2 = F.one_hot(a2, num_classes=N_ACTIONS)

            # concat actions to hidden state
            e1 = torch.cat((e1, a1), dim=0)
            e2 = torch.cat((e2, a2), dim=0)
            assert e1.shape == (self.hidden_dim+N_ACTIONS,), f"Expected {(self.hidden_dim+N_ACTIONS,)}, got {e1.shape}"

            key1 = self.lin2(torch.stack([e1, e2], dim=0))
            assert key1.shape == (2, self.hidden_dim), f"Expected {(2, self.hidden_dim)}, got {key1.shape}"

            # for QTRAN-alt
            joint_qs_prereduced.append(key1)

            combined_agents = torch.mean(key1, dim=0)
            # combined_agents = key1.view(1, -1)
            q = self.q_prime_net(combined_agents)
            joint_qs.append(q)
        
        joint_qs = torch.stack(joint_qs)
        joint_qs_prereduced = torch.stack(joint_qs_prereduced)

        joint_max_qs = []
        joint_qs_max_prereduced = []
        for a1, a2, e1, e2 in zip(a1_max, a2_max, h1, h2):
            a1 = F.one_hot(a1, num_classes=N_ACTIONS)
            a2 = F.one_hot(a2, num_classes=N_ACTIONS)

            # concat actions to hidden state
            e1 = torch.cat((e1, a1), dim=0)
            e2 = torch.cat((e2, a2), dim=0)

            key1_t = self.lin2(torch.stack([e1, e2], dim=0))
            assert key1_t.shape == (2, self.hidden_dim), f"Expected {(2, self.hidden_dim)}, got {key1_t.shape}"

            # for QTRAN-alt
            joint_qs_max_prereduced.append(key1_t)

            combined_agents = key1_t.mean(dim=0)
            # combined_agents = key1_t.view(1, -1)
            q = self.q_prime_net(combined_agents)
            joint_max_qs.append(q)

        joint_max_qs = torch.stack(joint_max_qs)
        joint_qs_max_prereduced = torch.stack(joint_qs_max_prereduced)

        assert joint_qs_prereduced.shape == (batch_size, 2, self.hidden_dim), f"Expected {(batch_size, 2, self.hidden_dim)}, got {joint_qs_prereduced.shape}"
        assert joint_qs_max_prereduced.shape == (batch_size, 2, self.hidden_dim), f"Expected {(batch_size, 2, self.hidden_dim)}, got {joint_qs_max_prereduced.shape}"

        averaged = torch.mean(joint_qs_prereduced, dim=1, keepdim=True)
        q_jt_alt = key2 + averaged - joint_qs_prereduced/2
        q_jt_alt = F.relu(self.lin3(q_jt_alt))
        q_jt_alt = self.lin4(q_jt_alt)

        averaged = torch.mean(joint_qs_max_prereduced, dim=1, keepdim=True)
        q_jt_max_alt = key2 + averaged - joint_qs_max_prereduced/2
        q_jt_max_alt = F.relu(self.lin3(q_jt_max_alt))
        q_jt_max_alt = self.lin4(q_jt_max_alt)

        return q1, q2, joint_qs, joint_max_qs, vtot, q_jt_alt, q_jt_max_alt

# MTX
mtx = [
    [8, -12, -12],
    [-12, 0, 0],
    [-12, 0, 0]
]

grid_size = N_ACTIONS

# # FROM CHATGPT
# def gaussian_2d(x, y, x0, y0, sigma):
#     return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
# x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# # Parameters for the Gaussian peaks
# peak1 = (5, 5)  # Coordinates of the first peak
# peak2 = (15, 15)  # Coordinates of the second peak
# sigma1 = 5  # Spread of the Gaussian
# sigma2 = 2  # Spread of the Gaussian
# # Generate the 2D array with two Gaussian peaks
# mtx = 7.5 * np.exp(-((x - peak1[0]) ** 2 + (y - peak1[1]) ** 2) / (2 * sigma1 ** 2))
# mtx += 10 * np.exp(-((x - peak2[0]) ** 2 + (y - peak2[1]) ** 2) / (2 * sigma2 ** 2))

# # Find min and max values in mtx
# min_val = np.min(mtx)
# max_val = np.max(mtx)
# # Scale and shift mtx to be in the range [-10, 10]
# mtx = 20 * (mtx - min_val) / (max_val - min_val) - 10

# FROM PAPER
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
f1 = 5 - ((15 - x)/3)**2 - ((5 - y)/3)**2
f2 = 10 - ((5 - x)/1)**2 - ((15 - y)/1)**2
mtx = np.maximum(f1, f2)
mtx = np.maximum(np.full_like(mtx, -10), mtx)

# f1 = np.maximum(np.full_like(f1, -5), f1)
# f2 = np.maximum(np.full_like(f2, -5), f2)
# mtx = f1 + f2

# normalize mtx
# mtx = (mtx - np.min(mtx)) / (np.max(mtx) - np.min(mtx))

plt.imshow(mtx, vmin=-10, vmax=10)
plt.show()

def initialize_weights(module):
    if isinstance(module, nn.Linear):  # Check if the layer is a Linear layer
        nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Initialize weights
        if module.bias is not None:  # Initialize bias, if exists
            nn.init.constant_(module.bias, 0.0)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
qtran = Qtran()
# qtran.apply(initialize_weights)
optimizer = optim.Adam(qtran.parameters(), lr=LR)
criterion = nn.MSELoss()

# TRAINING
for step in range(TRAINING_STEPS):
    # for each agent, choose action
    # epsilon = 1
    a1 = np.random.randint(N_ACTIONS)
    a2 = np.random.randint(N_ACTIONS)

    # get reward
    reward = mtx[a1][a2]

    # store in replay buffer
    replay_buffer.append((a1, a2, reward))

    if len(replay_buffer.buffer) < BATCH_SIZE:
        continue

    # sample from replay buffer
    batch = replay_buffer.sample(BATCH_SIZE)
    a1_batch = torch.tensor([x[0] for x in batch], dtype=torch.long)
    a2_batch = torch.tensor([x[1] for x in batch], dtype=torch.long)
    reward_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)

    # get q values
    q1, q2, joint_qs, joint_max_qs, vtot, q_jt_alt, q_jt_max_alt = qtran(torch.tensor([[[0,1], [1,0]]]*32, dtype=torch.float32), a1_batch, a2_batch)
    assert q1.shape == (BATCH_SIZE, N_ACTIONS), f"Expected {(BATCH_SIZE, N_ACTIONS)}, got {q1.shape}"
    assert q2.shape == (BATCH_SIZE, N_ACTIONS), f"Expected {(BATCH_SIZE, N_ACTIONS)}, got {q2.shape}"

    # zero out vtot
    vtot = torch.zeros_like(vtot)

    a1_max = torch.argmax(q1, dim=1)
    a2_max = torch.argmax(q2, dim=1)

    # y = reward_batch + GAMMA * q_tot[a1_max, a2_max]
    y = reward_batch

    q_prime_max = q1.gather(1, a1_max.unsqueeze(1)) + q2.gather(1, a2_max.unsqueeze(1))

    lv = torch.mean(vtot ** 2)

    # calculate loss
    if is_QTRAN_alt:
        # q_jt_alt: BATCH_SIZE x N_AGENTS x N_ACTIONS
        # sum(q_jt_alt[action])
        combined_onehot_actions = torch.cat([F.one_hot(a1_batch, num_classes=N_ACTIONS).unsqueeze(1), F.one_hot(a2_batch, num_classes=N_ACTIONS).unsqueeze(1)], dim=1)
        selected_Q = torch.sum(q_jt_alt * combined_onehot_actions, dim=2)

        assert selected_Q.shape == (BATCH_SIZE,2), f"Expected {(BATCH_SIZE,2)}, got {selected_Q.shape}"

        ltd = criterion(selected_Q, y.unsqueeze(1))

        assert a1_batch.shape == a1_max.shape, f"Expected {a1_batch.shape}, got {a1_max.shape}"

        combined_onehot_max_actions = torch.cat([F.one_hot(a1_max, num_classes=N_ACTIONS).unsqueeze(1), F.one_hot(a2_max, num_classes=N_ACTIONS).unsqueeze(1)], dim=1)
        max_Q = torch.sum(q_jt_max_alt * combined_onehot_max_actions, dim=2)

        max_Q = max_Q.detach()

        lopt = torch.mean((max_Q - q_prime_max) ** 2)

        # q1: (b, 2actions) -> (b x 1 x 2actions)
        # torch.cat((q1, q2), dim=1) -> (b x 2agents x 2actions)
        q_uj = torch.cat((q1.unsqueeze(1), q2.unsqueeze(1)), dim=1)
        assert q_uj.shape == (BATCH_SIZE, 2, N_ACTIONS), f"Expected {(BATCH_SIZE, 2, N_ACTIONS)}, got {q_uj.shape}"

        q_is = torch.cat((q1.gather(1, a1_batch.unsqueeze(1)), q2.gather(1, a2_batch.unsqueeze(1))), dim=1)
        q_is = q_is.view((-1, 2, 1))
        assert q_is.shape == (BATCH_SIZE, 2, 1), f"Expected {(BATCH_SIZE, 2, 1)}, got {q_is.shape}"

        q_prime_alt = q_uj - q_is + torch.sum(q_is, dim=1, keepdim=True).view((-1, 1, 1))

        lnopt_min = torch.mean(torch.min(q_prime_alt - q_jt_alt.detach(), dim=2).values ** 2)

        loss = ltd + LAMBDA * lopt + LAMBDA * lnopt_min
    else:
        ltd = criterion(joint_qs, y.unsqueeze(1))
        lopt = criterion(q_prime_max, joint_max_qs - vtot)
        lopt = torch.mean((q_prime_max - joint_max_qs + vtot) ** 2)
        
        q_prime_chosen = q1.gather(dim=1, index=a1_batch.unsqueeze(1)) + q2.gather(dim=1, index=a2_batch.unsqueeze(1))
        lnopt = torch.mean(torch.min(q_prime_chosen - joint_qs + vtot, torch.zeros_like(joint_qs)) ** 2)

        loss = ltd + LAMBDA * lopt + LAMBDA * lnopt
        # + LAMBDA * lv

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Qjt_array = np.zeros((N_ACTIONS, N_ACTIONS))
    q_prime_jt_array = np.zeros((N_ACTIONS, N_ACTIONS))

    # visualize for 21 x 21 mtx
    if len(mtx) == 21 and step % 500 == 0:
        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                q1, q2, joint_qs, joint_max_qs, vtot, q_jt_alt, q_jt_max_alt = qtran(torch.tensor([[[0,1], [1,0]]], dtype=torch.float32), torch.tensor([i]), torch.tensor([j]))

                combined_onehot_actions = torch.cat([F.one_hot(torch.tensor([i]), num_classes=N_ACTIONS).unsqueeze(1), F.one_hot(torch.tensor([j]), num_classes=N_ACTIONS).unsqueeze(1)], dim=1)
                selected_Q = torch.sum(q_jt_alt * combined_onehot_actions, dim=2)
                
                if is_QTRAN_alt:
                    Qjt_array[i][j] = torch.mean(selected_Q).detach().numpy()
                else:
                    Qjt_array[i][j] = joint_qs.detach().numpy()

        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                q_prime_jt_array[i][j] = (q1[0][i] + q2[0][j]).item()
            
        print("Qjt")
        plt.imshow(Qjt_array, vmin=-10, vmax=10)
        plt.show()

        print("Q'jt")
        plt.imshow(q_prime_jt_array, vmin=-10, vmax=10)
        plt.show()

    if step == TRAINING_STEPS - 1:
        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                q1, q2, joint_qs, joint_max_qs, vtot, q_jt_alt, q_jt_max_alt = qtran(torch.tensor([[[0,1], [1,0]]], dtype=torch.float32), torch.tensor([i]), torch.tensor([j]))

                combined_onehot_actions = torch.cat([F.one_hot(torch.tensor([i]), num_classes=N_ACTIONS).unsqueeze(1), F.one_hot(torch.tensor([j]), num_classes=N_ACTIONS).unsqueeze(1)], dim=1)
                selected_Q = torch.sum(q_jt_alt * combined_onehot_actions, dim=2)
                
                if is_QTRAN_alt:
                    Qjt_array[i][j] = torch.mean(selected_Q).detach().numpy()
                else:
                    Qjt_array[i][j] = joint_qs.detach().numpy()

        for i in range(N_ACTIONS):
            for j in range(N_ACTIONS):
                q1, q2, joint_qs, _, _, _, _ = qtran(torch.tensor([[[0,1], [1,0]]], dtype=torch.float32), torch.tensor([i]), torch.tensor([j]))
                q_prime_jt_array[i][j] = (q1[0][i] + q2[0][j]).item()

        plt.imshow(Qjt_array, vmin=-10, vmax=10)
        plt.savefig('Qjt.png', bbox_inches='tight')
        plt.imshow(q_prime_jt_array, vmin=-10, vmax=10)
        plt.savefig('Q\'jt.png', bbox_inches='tight')


    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)

    # arr = mtx
    # df = pd.DataFrame(arr.round(0).astype(int))
    # df


    # print qtot values
    if step % 100 == 0:
        if is_QTRAN_alt:
            logger.print(f"Step: {step}, Loss: {np.round(loss.item(), 3)}, Ltd: {np.round(ltd.item(), 3)}, Lopt: {np.round(lopt.item(), 3)}, Lnopt_min: {np.round(lnopt_min.item(), 3)}, Lv: {np.round(lv.item(), 3)}")
        else:
            logger.print(f"Step: {step}, Loss: {np.round(loss.item(), 3)}, Ltd: {np.round(ltd.item(), 3)}, Lopt: {np.round(lopt.item(), 3)}, Lnopt: {np.round(lnopt.item(), 3)}, Lv: {np.round(lv.item(), 3)}")
        logger.print("")

        # q1, q2, joint_qs, joint_max_qs, vtot = qtran(torch.tensor([1], dtype=torch.float32).unsqueeze(1), torch.tensor([0]), torch.tensor([0]))

        # # Q1, Q2, Q'jt
        # logger.print("Q1, Q2, Q'jt")
        # logger.print("\t", np.round(q2.detach().numpy()[0], 2))
        # for i in range(N_ACTIONS):
        #     q1_i = round(q1[0][i].item(), 2)
        #     logger.print(np.round(q1.detach().numpy()[0][i], 2), end=" ")
        #     for j in range(N_ACTIONS):
        #         q2_j = round(q2[0][j].item(), 2)
        #         logger.print(q1_i + q2_j, end=" ")
        #     logger.print("")
        # logger.print("")
        

        # # Qjt
        # logger.print("Qjt")
        # for i in range(N_ACTIONS):
        #     for j in range(N_ACTIONS):
        #         _, _, joint_qs, _, _ = qtran(torch.tensor([1], dtype=torch.float32).(1), torch.tensor([i]), torch.tensor([j]))

        #         logger.print(np.round(joint_qs.detach().squeeze().numpy(), 2), end=" ")
        #     logger.print("")

        # # print Q'jt - Qjt
        # logger.print("\nQ'jt - Qjt")
        # for i in range(N_ACTIONS):
        #     for j in range(N_ACTIONS):
        #         q_prime_i = round((q1[0][i] + q2[0][j]).item(), 2)
        #         _, _, joint_qs, _, _ = qtran(torch.tensor([1], dtype=torch.float32).unsqueeze(1), torch.tensor([i]), torch.tensor([j]))

        #         q_tot_i = np.round(joint_qs.detach().item(), 2)
        #         logger.print(round(q_prime_i - q_tot_i, 2), end=" ")
        #     logger.print("")

        # logger.print("")


# %%
