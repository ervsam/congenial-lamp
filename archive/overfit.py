# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import copy
import time
import yaml

from utils import *
from Environment import Environment
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from Trainer import Trainer

# set np seed
np.random.seed(0)

OVERFIT_TEST = 3

SIZE = 14
NUM_AGENTS = 7
OBSTACLE_DENSITY = 0.5
FOV = 7
WINDOW_SIZE = 3

BATCH_SIZE = 32
LAMBDA = 1
LR = 1e-4
BUFFER_SIZE = 10000
TRAIN_STEPS = 1e10
N_ACTIONS = 2

is_QTRAN_alt = True

# %%
logger = Logger("overfit.txt")

env = Environment(size_x=SIZE,
                  size_y=SIZE,
                  num_agents=NUM_AGENTS,
                  obstacle_density=OBSTACLE_DENSITY,
                  fov=FOV,
                  window_size=WINDOW_SIZE,
                  logger=logger,
                  grid_map_file=None,
                  starts=None, goals=None,
                  heuristic_map_file="heuristic_map.npy")

# buffer = ReplayBuffer(buffer_size=BUFFER_SIZE)
buffer = PrioritizedReplayBuffer(capacity=BUFFER_SIZE)

trainer = Trainer(LR, BATCH_SIZE, NUM_AGENTS, FOV, is_QTRAN_alt, LAMBDA, env)

throughput = []
losses = []

# %% TRAINING LOOP

if OVERFIT_TEST == 1:
    FIXED_START = [(11, 5), (11, 6), (11, 7), (13, 5)]
    FIXED_GOALS = [[(12, 3), (10, 2), (1, 2)],
                   [(11, 7), (13, 8), (1, 2)],
                   [(12, 5), (14, 6), (1, 2)],
                   [(11, 6), (9, 7), (1, 2)]]
elif OVERFIT_TEST == 2:
    FIXED_START = [(12, 3), (12, 4)]
    FIXED_GOALS = [[(14, 6), (14, 8)],
                   [(10, 2), (1, 2)],]
elif OVERFIT_TEST == 3:
    FIXED_START = [(11, 5), (11, 6), (11, 7), (13, 5), (4, 6), (4, 8), (4, 9)]
    FIXED_GOALS = [[(12, 3), (10, 2), (1, 2)],
                   [(11, 7), (13, 8), (1, 2)],
                   [(12, 5), (14, 6), (1, 2)],
                   [(11, 6), (9, 7), (1, 2)],
                   [(2, 5), (9, 7), (1, 2)],
                   [(2, 13), (9, 7), (1, 2)],
                   [(2, 14), (9, 7), (1, 2)],
                   ]

new_start, new_goals = env.starts, env.goals
steps = 0

while steps < TRAIN_STEPS:
    steps += 1
    timestep_start = time.time()
    logger.print("Step", steps, "\n")

    # FORCE START AND GOAL LOCATION
    if OVERFIT_TEST != 0:
        new_start, new_goals = copy.deepcopy(FIXED_START), copy.deepcopy(FIXED_GOALS)
        env.starts, env.goals = new_start, new_goals

    # GET OBSERVATION FOVS
    old_start, old_goals = copy.deepcopy(new_start), copy.deepcopy(new_goals)
    obs_fovs = env.get_obs()
    close_pairs = env.get_close_pairs()

    if close_pairs == []:
        logger.print("No close pairs, skipping instance\n")
        _, _, _, new_start, new_goals, throughput = step(env, logger, throughput, None, policy="random")
        continue

    groups = env.connected_edge_groups()

    # GET Q_VALS
    pair_enc, q_vals = trainer.q_net(obs_fovs.unsqueeze(0), [close_pairs])
    assert q_vals[0].shape == (len(close_pairs), 2)

    logger.print("Q_vals:")
    for pair, qval in zip(close_pairs, q_vals[0]):
        logger.print(pair, np.round(qval.detach().numpy().squeeze(), 3))
    logger.print()
    
    priorities, partial_prio, pred_value, new_start, new_goals, throughput = step(env, logger, throughput, q_vals, policy="random")

    if priorities is None:
        env.reset()
        new_start, new_goals = env.starts, env.goals
        continue

    logger.print("Priority ordering:", priorities, "\n")
    logger.print("Time to solve instance:", time.time()-timestep_start, "\n")

    # print Q'jt based on action taken
    q_prime = sum([q[a] for q, a in zip(q_vals[0], list(partial_prio.values()))])
    if len(groups) > 1:
        chosen_q = [q[a] for q, a in zip(q_vals[0], list(partial_prio.values()))]
        group_q_prime = []
        for group in groups:
            qprim = 0
            for pair, q in zip(close_pairs, chosen_q):
                if pair in group:
                    qprim += q
            group_q_prime.append(qprim)
        q_prime = group_q_prime

    logger.print("Q'jt:", [round(q.item(), 3) for q in q_prime])

    # print vtot
    group_pair_enc = []
    for group in groups:
        subgroup_pair_enc = []
        for pair, enc in zip(close_pairs, pair_enc[0]):
            if pair in group:
                subgroup_pair_enc.append(enc)
        group_pair_enc.append(torch.stack(subgroup_pair_enc))
    # batch_pair_enc : 'groups' x 'n_pairs' x h*2
    vtot = torch.stack(trainer.vnet(group_pair_enc))
    logger.print("Vtot:", [round(q.item(), 3) for q in vtot])

    actions = torch.stack(list(partial_prio.values()))
    # turn actions to one hot encoding and append to pair_enc
    tmp = []
    for pair_enc in pair_enc:
        tmp += pair_enc
    pair_enc = torch.stack(tmp)
    batch_onehot_actions = F.one_hot(actions, num_classes=2).view(-1, 2)
    batch_pair_enc_action = torch.concat([pair_enc, batch_onehot_actions], dim=1)

    ############ if using Q-JOINT ############
    # q_jt, q_jt_alt = trainer.qjoint_net(pair_enc, batch_pair_enc_action, [close_pairs], [groups], [len(partial_prio)])
    # if is_QTRAN_alt:
    #     group_onehot = []
    #     for group in groups:
    #         subgroup_actions = []
    #         for pair, act in zip(close_pairs, batch_onehot_actions):
    #             if pair in group:
    #                 subgroup_actions.append(act)
    #         group_onehot.append(torch.stack(subgroup_actions))
    
    #     selected_Qs = []
    #     for q, act in zip(q_jt_alt, group_onehot):
    #         selected_Qs.append(list((torch.sum((q * act), dim=1)).detach().numpy()))
    #     # selected_Q = torch.sum((q_jt_alt * batch_onehot_actions), dim=1)

    #     logger.print("Q-joint predicted:")
    #     for i in selected_Qs:
    #         logger.print([round(j, 3) for j in i])

    #     # print Q'jt - Qjt
    #     logger.print("Q'jt - Qjt:")
    #     for i, group_q in enumerate(selected_Qs):
    #         logger.print([round(q_prime[i].item() - j, 3) for j in group_q])
    #     logger.print()
    # else:
    #     logger.print("Q-joint predicted:", [round(i.item(), 2) for i in q_jt[0]], " = ", q_jt[0].sum().item())

    #     # print Q'jt - Qjt
    #     logger.print("Q'jt - Qjt:", q_prime.item() - q_jt[0].sum().item(), "\n")

    max_delay = 6
    global_reward = sum([-x for x in env.get_delays()])
    # global_reward = 1 / (sum(env.get_delays()) + 1)

    local_rewards = []
    group_agents = []
    delays = env.get_delays()
    for group in groups:
        set_s = set()
        for pair in group:
            set_s.update(pair)
        group_agents.append(list(set_s))
    for group in group_agents:
        local_rewards.append(sum([-delays[agent] for agent in group]))
        # local_rewards.append(1 / (sum([delays[agent] for agent in group]) + 1))

    # global_reward = sum([-x/max_delay for x in env.get_delays()])/env.num_agents
    # global_reward = 1 / (sum(env.get_delays()) + 1)
    logger.print("Global Reward:", global_reward, "\n")
    if len(groups) > 1:
        logger.print("Multiple groups detected")
    logger.print("Groups:", group_agents)
    logger.print("Delays:", delays)
    logger.print("Local Rewards:", [round(i, 3) for i in local_rewards], "\n")

    buffer.insert(obs_fovs, partial_prio, global_reward, local_rewards, groups, old_start, old_goals)

    if steps % 100 == 0:
        plot(losses, ylabel="Total Loss", xlabel="Steps", filename="loss_plot_overfit.png")
        # plot(throughput, ylabel="Throughput", xlabel="Steps", filename="throughput_plot.png")
    # save model
    if steps % 5000 == 0:
        torch.save(trainer.q_net.state_dict(), f'q_net_model/q_net_{steps}.pth')
        # torch.save(trainer.qjoint_net.state_dict(), f'qjoint_net_{i}.pth')
        # torch.save(trainer.vnet.state_dict(), f'vnet_{i}.pth')

    # OPTIMIZE
    print("buffer size:", len(buffer))
    if len(buffer) > BATCH_SIZE:
        is_prioritized = isinstance(buffer, PrioritizedReplayBuffer)
        if is_prioritized:

            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals, indices = buffer.sample(BATCH_SIZE)
            print("indices:", indices)
        else:
            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups = buffer.sample(BATCH_SIZE)

        loss, ltd, lopt, lnopt, td_errors = trainer.optimize(batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals)
        if is_QTRAN_alt:
            logger.print("Loss:", loss, "LTD:", ltd,
                        "LOPT:", lopt, "LNOPT-min:", lnopt)
        else:
            logger.print("Loss:", loss, "LTD:", ltd,
                        "LOPT:", lopt, "LNOPT:", lnopt)
        losses.append(loss)

        if is_prioritized:
            assert len(indices) == len(td_errors)
            buffer.update_priorities(indices, td_errors)

    logger.print(
        "____________________________________________________________________________")
    
    if steps % 25000 == 0:
        logger.set_filename(f"overfit_{steps}.txt")



# %%
# given array of paths, visualize the path of all the agents in a gif, where each frame is the agent taking its step

# def save_paths_as_gif(grid, paths, goals, filename='path_animation.gif', with_lines=False):
#     '''
#     Visualize the path(s) overlayed on the grid map and create a GIF from each time step

#     Params
#     ======
#         grid (numpy array): grid map
#         paths (list): list of paths, where each path is a list of (x, y) coordinates

#     Returns
#     =======
#         None
#     '''

#     colors = [np.random.rand(3) for i in range(1000)]
#     colors = [plt.cm.hsv(i / env.num_agents) for i in range(env.num_agents)]


#     fig, ax = plt.subplots()

#     def update_frame(i):
#         ax.clear()

#         # Plot the grid map
#         ax.imshow(~grid, cmap='gray')

#         # Plot each path with a different color
#         for idx, (path, goal) in enumerate(zip(paths, goals)):
#             path_x, path_y = zip(*path[:i+1])
#             if with_lines == 'partial':
#                 ax.plot(path_y[-2:], path_x[-2:], color=colors[idx])
#             elif with_lines:
#                 ax.plot(path_y, path_x, color=colors[idx])

#             # start_x, start_y = path[0]
#             ax.scatter(path_y[-1], path_x[-1], marker='o', color=colors[idx])

#             # # Plot 'X' at the end point
#             # end_x, end_y = path[-1]
#             # ax.scatter(end_y, end_x, marker='x', color=colors[idx])

#             # Plot the goal
#             for t, g in enumerate(goal):
#                 # plt.text(g[1], g[0], str(t+1), color=colors[idx])
#                 ax.text(g[1], g[0], str(t+1), color=colors[idx])

#         # Set the aspect ratio
#         ax.set_aspect('equal')

#     # Create the animation
#     anim = animation.FuncAnimation(
#         fig, update_frame, frames=max(len(i) for i in paths), interval=800)
#     anim.save(filename, writer='pillow')

#     plt.close()


# # visualize_paths(grid_map, paths, start, goal, 'path.gif')
# save_paths_as_gif(env.grid_map, [[i[:2] for i in path]
#                   for path in env.paths], env._old_goals, filename='path.gif', with_lines=True)

