# %% IMPORTS & CONFIGURATION
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
import os
import shutil

from torchviz import make_dot

import torch, pickle, pathlib, numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import *                   # Custom utility functions and classes
from Environment import Environment    # Environment setup class
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from Trainer import Trainer            # Trainer class for Q-networks / Qjoint networks
from Model import Encoder

# Set global numpy seed for reproducibility
np.random.seed(0)

config_name = "warehouse_2"

# Load configuration file
file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(file_path, "r") as file:
    config_file = yaml.safe_load(file)

config = config_file[config_name]

# Environment Configuration
env_config = config["environment"]

# Training Configuration
train_config   = config["training"]
OVERFIT_TEST   = train_config["OVERFIT_TEST"]
BATCH_SIZE     = train_config["BATCH_SIZE"]
LAMBDA         = train_config["LAMBDA"]
LR             = float(train_config["LR"])
BUFFER_SIZE    = train_config["BUFFER_SIZE"]
TRAIN_STEPS    = int(float(train_config["TRAIN_STEPS"]))  # Convert to int for loops
N_ACTIONS      = train_config["N_ACTIONS"]
DEVICE         = train_config["DEVICE"]
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

USE_PBS = train_config["USE_PBS"]
USE_CURRICULUM = train_config["USE_CURRICULUM"]

# Paths Configuration
paths_config      = config["paths"]
GRID_MAP_FILE     = paths_config["map_file"]
HEURISTIC_MAP_FILE= paths_config["heur_file"]
DIR               = config["root"] + paths_config["results"]
saved_model_path  = os.path.join(DIR, 'q_net_model/')
log_path          = os.path.join(DIR, "log/")

# Clean up previous directories if they exist and create new ones.
if os.path.exists(saved_model_path):
    shutil.rmtree(saved_model_path)
os.makedirs(saved_model_path)
if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)

# Use QTRAN alternative version flag
is_QTRAN_alt = True

is_logged = {
    "Q-vals": False,
    "difference": False,
}

# %% LOGGER & ENVIRONMENT INITIALIZATION
logger = Logger(os.path.join(log_path, "log.txt"))
logger.print('[torch] using', device)

buffer = PrioritizedReplayBuffer(capacity=BUFFER_SIZE, freeze=False)
# buffer = ReplayBuffer(buffer_size=BUFFER_SIZE)

throughput = []
losses = []
ltds = []
lopts = []
lnopts = []

# %% SET FIXED START/GOALS FOR OVERFIT TEST IF APPLICABLE
if OVERFIT_TEST == 1:
    FIXED_START = [(11, 5), (11, 6), (11, 7), (13, 5)]
    FIXED_GOALS = [[(12, 3), (10, 2), (1, 2)],
                   [(11, 7), (13, 8), (1, 2)],
                   [(12, 5), (14, 6), (1, 2)],
                   [(11, 6), (9, 7), (1, 2)]]
elif OVERFIT_TEST == 2:
    FIXED_START = [(12, 3), (12, 4)]
    FIXED_GOALS = [[(14, 6), (14, 8)],
                   [(10, 2), (1, 2)]]
elif OVERFIT_TEST == 3:
    FIXED_START = [(11, 5), (11, 6), (11, 7), (13, 5), (4, 6), (4, 8), (4, 9)]
    FIXED_GOALS = [[(12, 3), (10, 2), (1, 2)],
                   [(11, 7), (13, 8), (1, 2)],
                   [(12, 5), (14, 6), (1, 2)],
                   [(11, 6), (9, 7), (1, 2)],
                   [(2, 5), (9, 7), (1, 2)],
                   [(2, 13), (9, 7), (1, 2)],
                   [(2, 14), (9, 7), (1, 2)]]
    
# ----------------------------------------------------------------------
# dev_buffer.py  (a *very* small wrapper around your existing buffer)
# ----------------------------------------------------------------------
class DevBuffer:
    """Holds a frozen set of (s,a,r) tuples for validation only."""
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []          # list of tuples same format as main buffer

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)   # push until full; never overwritten

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals = zip(*[self.buffer[i] for i in indices])
        return obs_fovs, partial_prio, global_reward, local_rewards, groups, starts, goals

    def __len__(self):
        return len(self.buffer)

dev_buffer = DevBuffer()

def embed(Z, method="pca"):
    if method == "pca":
        return PCA(n_components=2).fit_transform(Z)
    if method == "tsne":
        return TSNE(n_components=2,
                    perplexity=50,
                    init="pca",
                    n_iter=3000,
                    learning_rate="auto").fit_transform(Z)
    
def encode_batch(batch_obs, encoder):
    """
    Args
    ----
    batch_obs : list of tensors, len = batch_size
                each tensor  (n_agents, C, H, W)

    Returns
    -------
    latents : (N_total_agents, LATENT_DIM)  torch tensor
    """
    all_img  = []
    all_coord= []
    for obs in batch_obs:
        # split coords & image channels exactly as in q_net.forward
        coords = torch.tensor(
            list(zip(obs[:, -1, 0, 0], obs[:, -1, 0, 1])),
            dtype=torch.float32)
        imgs   = obs[:, :-1]                  # drop coord-channel

        all_img.append(imgs)
        all_coord.append(coords)

    imgs   = torch.cat(all_img,   dim=0)
    coords = torch.cat(all_coord, dim=0)
    with torch.no_grad():
        z = encoder(imgs, coords)
    return z

def unpack_buffer(buffer, encoder, max_episodes=600, rare_thresh=-0.6):
    """Return latent vectors + labels."""
    latent, domain, rarity = [], [], []
    for k, sample in enumerate(buffer.buffer[:max_episodes]):
        obs, _, _, local_rewds, *_ = sample

        z = encode_batch([obs], encoder)               # (n_agents, D)
        latent.append(z)

        # label per-agent the rarity of its *episode*
        r = min(local_rewds)                  # worst group reward
        rare_flag = int(r <= rare_thresh)
        rarity.extend([rare_flag] * len(z))
        domain.extend( [-1] * len(z) )   # overwritten later

    return torch.cat(latent), np.array(domain), np.array(rarity)

# %% MAIN TRAINING LOOP

# --------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------- #
def anneal(val0: float, step: int, k: float, floor: float = .1) -> float:
    """generic 1/(1+k·t) decay"""
    return max(val0 / (1 + k * step), floor)

def make_env(num_agents: int) -> Environment:
    env_cfg = {**env_config, "NUM_AGENTS": num_agents}
    return Environment(env_cfg, logger=logger, grid_map_file=GRID_MAP_FILE, heuristic_map_file=HEURISTIC_MAP_FILE)

# --------------------------------------------------------------------- #
#  MAIN LOOP
# --------------------------------------------------------------------- #
env          = make_env(env_config["NUM_AGENTS"])
trainer      = Trainer(LR, BATCH_SIZE, is_QTRAN_alt, LAMBDA, env, device).to(device)

steps        = 0
pbs_steps    = 0
curriculum_loss  : list[float] = []

new_start, new_goals = env.starts, env.goals
epsilon_start = 0.8
PBS_epsilon_start = 0
pbs_epsilon = 0
decay_rate = 1e-3

while steps < TRAIN_STEPS:
    # ---------- curriculum -------------------------------------------------
    if (USE_CURRICULUM and len(curriculum_loss) > 1000 and np.mean(curriculum_loss[-100:]) < 3):
        torch.save(trainer.q_net.state_dict(), f"{saved_model_path}/q_{steps}.pth")
        env = make_env(env.num_agents + 2).to(device)
        # trainer.rebind_env(env)
        trainer.env = copy.deepcopy(env)
        trainer.num_agents = env.num_agents
        curriculum_loss.clear()
        pbs_steps = 0
        logger.print(f"Moving on to {env_config['NUM_AGENTS']} agents", "\n")


    # ---------- episode set-up --------------------------------------------
    steps += 1
    epsilon     = anneal(epsilon_start, steps, decay_rate)
    if USE_PBS:
        pbs_epsilon = anneal(PBS_epsilon_start, pbs_steps, decay_rate) if USE_PBS else 0
    pbs_steps  += 1

    timestep_start = time.time()
    logger.print("Step", steps, "\n")

    ############################# GET OBSERVATION ####################################
    
    if OVERFIT_TEST:
        new_start, new_goals = copy.deepcopy(FIXED_START), copy.deepcopy(FIXED_GOALS)
        env.starts, env.goals = new_start, new_goals
        env.DHC_heur = env._get_DHC_heur()

    old_start, old_goals = copy.deepcopy(new_start), copy.deepcopy(new_goals)
    obs_fovs = env.get_obs().to(device)
    close_pairs = env.get_close_pairs()

    if close_pairs == []:
        logger.print("No close pairs, skipping instance\n")
        _, _, _, new_start, new_goals, throughput = step(env, logger, throughput, None, policy="random", pbs_epsilon=0)
        continue

    groups = env.connected_edge_groups()

    # ---------- inference (no grad) ---------------------------------------
    with torch.no_grad():
        trainer.eval_mode()

        # ---- individual Q-values --------------------------------------

        pair_enc, q_vals = trainer.q_net(obs_fovs.unsqueeze(0), [close_pairs])
        assert q_vals[0].shape == (len(close_pairs), 2)

        if is_logged["Q-vals"]:
            logger.print("Q-vals:")
            for pair, qval in zip(close_pairs, q_vals[0]):
                logger.print(pair, np.round(qval.detach().cpu().squeeze().numpy(), 3))
            logger.print()

        ######################### ENV STEP ###############################################
        #  take environment step with ε-greedy / PBS policy
        priorities, partial_prio, *_ = step(
            env, logger, throughput, [qval.cpu() for qval in q_vals],
            "epsilon_greedy", epsilon, pbs_epsilon,
            obs_fovs, buffer, old_start, old_goals)

        if priorities is None:      # infeasible → new episode
            env.reset()
            new_start, new_goals = env.starts, env.goals
            continue

        ################################ INSERT (s, a, r) TO BUFFER ########w############################

        delays = env.get_delays()
        global_reward = -sum(delays)

        group_agents = []
        for group in groups:
            set_s = set()
            for pair in group:
                set_s.update(pair)
            group_agents.append(list(set_s))

        local_rewards = [10 * (sum([-delays[agent] for agent in group]) / len(group)) - 0.2 for group in group_agents] # 10 *ave reward
        # local_rewards.append(sum([-delays[agent] for agent in group]))
        # local_rewards.append(env.num_agents * sum([-delays[agent] for agent in group]) / len(group))
        # local_rewards.append(10 / (sum([delays[agent] for agent in group]) + 1))
        # local_rewards.append((1.1 ** -(sum([delays[agent] for agent in group])) * 10))

        # --- 20 % of new samples go into the (frozen) dev set (store per-group) -------------
        if np.random.rand() < 0.20:
            # dev_buffer.add((obs_fovs, partial_prio, global_reward, local_rewards, groups, old_start, old_goals))
            for group, local_reward in zip(groups, local_rewards):
                group_partial_prio = {pair: partial_prio[pair] for pair in group}
                dev_buffer.add((obs_fovs, group_partial_prio, global_reward, [local_reward], [group], old_start, old_goals))
        else:
            for group, local_reward in zip(groups, local_rewards):
                group_partial_prio = {pair: partial_prio[pair] for pair in group}
                buffer.insert(obs_fovs, group_partial_prio, global_reward, [local_reward], [group], old_start, old_goals)
            # buffer.insert(obs_fovs, partial_prio, global_reward, local_rewards, groups, old_start, old_goals)

        logger.print("Priority ordering:", priorities, "\n")
        logger.print("Time to solve instance:", time.time()-timestep_start, "\n")

        # Calculate Q'jt based on action taken (group-based sum)
        q_prime = [sum([q[a] for q, a in zip(q_vals[0], list(partial_prio.values()))])]
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

        # Compute Vtot per group.
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
        # logger.print("Actions:", actions)
        # turn actions to one hot encoding and append to pair_enc
        tmp = []
        for enc in pair_enc:
            tmp += enc
        pair_enc = torch.stack(tmp)
        batch_onehot_actions = F.one_hot(actions, num_classes=2).view(-1, 2).float()
        batch_onehot_actions = batch_onehot_actions.to(device)
        batch_pair_enc_action = torch.concat([pair_enc, batch_onehot_actions], dim=1)

        ############ if using Q-JOINT ############
        q_jt, q_jt_alt = trainer.qjoint_net(pair_enc, batch_pair_enc_action.to(device), [close_pairs], [groups], [len(partial_prio)])

        if is_QTRAN_alt:
            # logger.print("Q-joint alt predicted:")
            # for i in q_jt_alt:
            #     logger.print(i.detach().numpy())
            group_onehot = []
            for group in groups:
                subgroup_actions = []
                for pair, act in zip(close_pairs, batch_onehot_actions):
                    if pair in group:
                        subgroup_actions.append(act)
                group_onehot.append(torch.stack(subgroup_actions))

            # logger.print("Group one-hot actions:")
            # for i in group_onehot:
            #     logger.print(i.detach().numpy())
        
            selected_Qs = []
            for q, act in zip(q_jt_alt, group_onehot):
                selected_Qs.append(list((torch.sum((q * act), dim=1)).detach().cpu().numpy()))

            logger.print("Q-joint chosen:")
            for i in selected_Qs:
                logger.print([round(j, 3) for j in i])

            # print Q'jt - Qjt
            if is_logged["difference"]:
                logger.print("Q'jt - Qjt:")
                for i, group_q in enumerate(selected_Qs):
                    logger.print([round(q_prime[i].item() - j, 3) for j in group_q])
            logger.print()
        else:
            logger.print("Q-joint predicted:", [round(i.item(), 2) for i in q_jt[0]], " = ", q_jt[0].sum().item())
            logger.print("Q'jt - Qjt:", q_prime.item() - q_jt[0].sum().item(), "\n")

        logger.print("Global Reward:", global_reward)
        if len(groups) > 1:
            logger.print("Multiple groups detected")
        logger.print("Groups:", group_agents)
        logger.print("Delays:", delays)
        logger.print("Local Rewards:", [round(i, 3) for i in local_rewards], "\n")


    if steps % 250 == 0:
        plot(losses, ylabel="Total Loss", xlabel="Steps", filename=DIR + "loss_plot.png")
        # plot(throughput, ylabel="Throughput", xlabel="Steps", filename="throughput_plot.png")

        # plot ltds, lopts, lnopts in one plot
        plt.figure()
        plt.plot(ltds, label="LTD", color='red')
        plt.plot(lopts, label="LOPT", color='green')
        plt.plot(lnopts, label="LNOPT", color='blue')
        plt.yscale('log')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Components")
        plt.savefig(DIR+"loss_components_plot.png")
        plt.close()


    # save model
    if steps % 5000 == 0:
        torch.save(trainer.q_net.state_dict(), saved_model_path+f"q_net_{steps}.pth")
        # torch.save(trainer.qjoint_net.state_dict(), f'qjoint_net_{i}.pth')
        # torch.save(trainer.vnet.state_dict(), f'vnet_{i}.pth')

    ########################################### OPTIMIZE ############################################
    if len(buffer) >= BATCH_SIZE:
        is_prioritized = isinstance(buffer, PrioritizedReplayBuffer)
        if is_prioritized:
            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals, indices, weights = buffer.sample(BATCH_SIZE)
        else:
            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups = buffer.sample(BATCH_SIZE)

        trainer.q_net.train()
        trainer.qjoint_net.train()
        loss, ltd, lopt, lnopt, td_errors = trainer.optimize(batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals, logger=logger, weights=weights)
        logger.print("training in", next(trainer.q_net.parameters()).device)      # should print 'cuda:0'

        if is_QTRAN_alt:
            logger.print("Loss:", loss, "LTD:", ltd,
                        "LOPT:", lopt, "LNOPT-min:", lnopt)
        else:
            logger.print("Loss:", loss, "LTD:", ltd,
                        "LOPT:", lopt, "LNOPT:", lnopt)
        losses.append(loss)
        ltds.append(ltd)
        lopts.append(lopt)
        lnopts.append(lnopt)

        curriculum_loss.append(loss)

        if is_prioritized:
            assert len(indices) == len(td_errors), f"Length of indices {len(indices)} and td_errors {len(td_errors)} do not match"
            buffer.update_priorities(indices, td_errors)

    if len(dev_buffer) >= BATCH_SIZE and steps % 250 == 0:
        with torch.no_grad():
            trainer.q_net.eval()
            trainer.qjoint_net.eval()
            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals = dev_buffer.sample(BATCH_SIZE)
            
            # move to device
            batch_obs = [obs.to(device) for obs in batch_obs]
            
            num_pairs_in_group = [] # len: batch_size * 'n_groups'
            for groups in batch_groups:
                for group in groups:
                    num_pairs_in_group.append(len(group))

            batch_close_pairs = [list(pp.keys()) for pp in batch_partial_prio]
            batch_pair_enc, batch_q_vals = trainer.q_net(batch_obs, batch_close_pairs)
            tmp = []
            for pair_enc in batch_pair_enc:
                tmp += pair_enc
            batch_pair_enc = torch.stack(tmp)

            batch_actions = []
            for pp in batch_partial_prio:
                batch_actions += list(pp.values())
            batch_actions = torch.stack(batch_actions)
            batch_onehot_actions = F.one_hot(batch_actions, num_classes=2).squeeze().to(device)
            batch_pair_enc_action = torch.concat([batch_pair_enc, batch_onehot_actions], dim=1)

            num_n_pairs = []
            for pp in batch_partial_prio:
                num_n_pairs.append(len(pp.values()))

            q_jt, q_jt_alt = trainer.qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)

            tmp = []
            for t in q_jt_alt:
                tmp += t
            q_jt_alt = torch.stack(tmp)

            selected_Q = torch.sum(q_jt_alt * batch_onehot_actions, dim=1).unsqueeze(1) # b x 1

            logger.print("------------------------ DEV BUFFER ------------------------")

            logger.print("Q-joint predicted, local rewards:")

            selected_Q_each_pair = [round(i, 3) for i in selected_Q.squeeze().detach().cpu().numpy()]

            group_Q = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_Q.append((selected_Q_each_pair[cursor:cursor+n]))     # tensor with grad
                cursor += n

            episode_Q = []                                           # length = batch_size
            cursor = 0
            for groups in batch_groups:                                # list-of-lists
                k = len(groups)
                episode_Q.append(group_Q[cursor:cursor+k])  # tensor with grad
                cursor += k

            for q, r in zip(episode_Q, batch_local_rewards):
                logger.print(q, "---------------------", r)
            
            tmp = []
            for r in batch_local_rewards:
                tmp += r
            batch_local_rewards = torch.tensor(tmp, dtype=torch.float32, device=device)

            # expand
            tmp = []
            for idx, num in enumerate(num_pairs_in_group):
                tmp += [batch_local_rewards[idx]] * num
            batch_local_rewards = torch.stack(tmp)

            ltd_raw = F.mse_loss(selected_Q, batch_local_rewards.unsqueeze(1), reduction='none')

            dup_factor = torch.tensor(num_pairs_in_group, device=device)
            norm = torch.tensor(dup_factor).repeat_interleave(dup_factor)
            ltd_pairs = ltd_raw.squeeze() / norm          # both (N,)
            ltd_pairs = ltd_pairs.unsqueeze(1) 

            # --- aggregate to group level -------------------------------------------
            group_ltd = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_ltd.append(ltd_pairs[cursor:cursor+n].sum())     # tensor with grad
                cursor += n
            group_ltd = torch.stack(group_ltd)                         # (G,) keeps grad

            weights = torch.as_tensor(weights, device=device, dtype=group_ltd.dtype)

            exp_w = torch.repeat_interleave(weights, torch.as_tensor([len(g) for g in batch_groups], device=device))

            # --- aggregate to episode level -------------------------------------------
            # episode_ltd = []                                           # length = batch_size
            # cursor = 0
            # for groups in batch_groups:                                # list-of-lists
            #     k = len(groups)
            #     episode_ltd.append(group_ltd[cursor:cursor+k].mean())  # tensor with grad
            #     cursor += k
            # episode_ltd = torch.stack(episode_ltd)                     # (B,) keeps grad

            logger.print("DEV LTD:", (exp_w * group_ltd).mean())

            def bucket_error(pred, target):
                buckets = {}
                for p, t in zip(pred, target):
                    if round(float(t), 2) not in buckets:
                        buckets[round(float(t), 2)] = []
                    buckets[round(float(t), 2)].append(abs(p-t))
                return {k: np.mean(v) if v else 0 for k,v in buckets.items()}

            # after every dev pass
            errs = bucket_error(selected_Q.detach().cpu().numpy(), batch_local_rewards.detach().cpu().numpy())
            logger.print(errs)

            # encoder = trainer.q_net.encoder               # shortcut
            # encoder.eval()

            # # ---- collect training set -------------------------------------------------
            # train_lat, train_dom, train_rar = unpack_buffer(buffer, encoder)   # main PER buffer
            # train_dom[:] = 0                                          # 0 = train

            # # ---- collect dev set ------------------------------------------------------
            # dev_lat, dev_dom, dev_rar = unpack_buffer(dev_buffer, encoder)
            # dev_dom[:] = 1                                            # 1 = dev

            # Z       = torch.cat([train_lat, dev_lat]).numpy()         # (N, D)
            # domain  = np.concatenate([train_dom, dev_dom])
            # rarity  = np.concatenate([train_rar, dev_rar])

            # # logger.print("domain", domain)
            # # logger.print("rarity", rarity)

            # for method in ("pca", "tsne"):
            #     XY = embed(Z, method)
            #     # logger.print("XY", XY)
            #     # logger.print("XY shape", XY.shape)
                
            #     plt.figure(figsize=(6,5))

            #     # --- colour code -------------------------------------------------------
            #     colour = ["b", "r"]       # 0=train,1=dev
            #     marker = ["o", "x"]                       # 0=common,1=rare

            #     for d in (0,1):
            #         for r in (0,1):
            #             idx = np.where((domain==d) & (rarity==r))[0]
            #             # logger.print(XY[idx,0], XY[idx,1])
            #             plt.scatter(XY[idx,0], XY[idx,1],
            #                         c=colour[d],
            #                         marker=marker[r],
            #                         s=12, alpha=0.6,
            #                         label=f'{"train" if d==0 else "dev"} '
            #                             f'{"rare" if r==1 else "common"}')

            #     plt.title(f"{method.upper()} of encoder features")
            #     plt.legend(markerscale=1.5, fontsize=8)
            #     plt.tight_layout()
            #     plt.savefig(f"latent_{method}.png", dpi=180)
            #     plt.close()

            # encoder.train()

            # y_is_rare = (batch_local_rewards.abs() > 0.5).float()      # 0 or 1 target
            # logit   = trainer.qjoint_net.rare_head(batch_pair_enc).squeeze()

            # prob_rare = torch.sigmoid(logit)
            # pred      = (prob_rare > 0.5).float()
            # acc       = (pred == y_is_rare).float().mean()
            # rare_bce = F.binary_cross_entropy_with_logits(logit, y_is_rare)
            # logger.print(f'aux-BCE: {rare_bce.item():.4f} | aux-acc: {acc.item():.3f}')


    logger.print("____________________________________________________________________________")
    
    if steps % 25000 == 0:
        logger.set_filename(log_path + f"log_{steps}.txt")

    if steps % 500 == 0:
        buffer.save_to_file(DIR + "buffer.csv")

    # check prediction of data in buffer
    if steps % 100 == 0:
        with torch.no_grad():
            batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals, indices, weights = buffer.sample(BATCH_SIZE)

            # move to device
            batch_obs = [obs.to(device) for obs in batch_obs]

            num_pairs_in_group = [] # len: batch_size * 'n_groups'
            for groups in batch_groups:
                for group in groups:
                    num_pairs_in_group.append(len(group))

            batch_close_pairs = [list(pp.keys()) for pp in batch_partial_prio]
            batch_pair_enc, batch_q_vals = trainer.q_net(batch_obs, batch_close_pairs)
            tmp = []
            for pair_enc in batch_pair_enc:
                tmp += pair_enc
            batch_pair_enc = torch.stack(tmp)

            batch_actions = []
            for pp in batch_partial_prio:
                batch_actions += list(pp.values())
            batch_actions = torch.stack(batch_actions)
            batch_onehot_actions = F.one_hot(batch_actions, num_classes=2).squeeze().to(device)
            batch_pair_enc_action = torch.concat([batch_pair_enc, batch_onehot_actions], dim=1)

            num_n_pairs = []
            for pp in batch_partial_prio:
                num_n_pairs.append(len(pp.values()))

            q_jt, q_jt_alt = trainer.qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)

            tmp = []
            for t in q_jt_alt:
                tmp += t
            q_jt_alt = torch.stack(tmp)

            selected_Q = torch.sum(q_jt_alt * batch_onehot_actions, dim=1).unsqueeze(1) # b x 1

            logger.print("------------------------ checkpoint ------------------------")

            logger.print("Q-joint predicted, local rewards:")

            selected_Q_each_pair = [round(i, 3) for i in selected_Q.squeeze().detach().cpu().numpy()]

            group_Q = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_Q.append((selected_Q_each_pair[cursor:cursor+n]))     # tensor with grad
                cursor += n

            episode_Q = []                                           # length = batch_size
            cursor = 0
            for groups in batch_groups:                                # list-of-lists
                k = len(groups)
                episode_Q.append(group_Q[cursor:cursor+k])  # tensor with grad
                cursor += k

            for q, r in zip(episode_Q, batch_local_rewards):
                logger.print(q, "---------------------", [round(i, 3) for i in r])


# %%
# given array of paths, visualize the path of all the agents in a gif, where each frame is the agent taking its step

def save_paths_as_gif(grid, paths, goals, filename='path_animation.gif', with_lines=False):
    '''
    Visualize the path(s) overlayed on the grid map and create a GIF from each time step

    Params
    ======
        grid (numpy array): grid map
        paths (list): list of paths, where each path is a list of (x, y) coordinates

    Returns
    =======
        None
    '''

    colors = [np.random.rand(3) for i in range(1000)]
    colors = [plt.cm.hsv(i / env.num_agents) for i in range(env.num_agents)]


    fig, ax = plt.subplots()

    def update_frame(i):
        ax.clear()

        # Plot the grid map
        ax.imshow(~grid, cmap='gray')

        # Plot each path with a different color
        for idx, (path, goal) in enumerate(zip(paths, goals)):
            path_x, path_y = zip(*path[:i+1])
            if with_lines == 'partial':
                ax.plot(path_y[-2:], path_x[-2:], color=colors[idx])
            elif with_lines:
                ax.plot(path_y, path_x, color=colors[idx])

            # start_x, start_y = path[0]
            ax.scatter(path_y[-1], path_x[-1], marker='o', color=colors[idx])

            # # Plot 'X' at the end point
            # end_x, end_y = path[-1]
            # ax.scatter(end_y, end_x, marker='x', color=colors[idx])

            # Plot the goal
            for t, g in enumerate(goal):
                # plt.text(g[1], g[0], str(t+1), color=colors[idx])
                ax.text(g[1], g[0], str(t+1), color=colors[idx])

        # Set the aspect ratio
        ax.set_aspect('equal')

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=max(len(i) for i in paths), interval=800)
    anim.save(filename, writer='pillow')

    plt.close()


# visualize_paths(grid_map, paths, start, goal, 'path.gif')
save_paths_as_gif(env.grid_map, [[i[:2] for i in path]
                  for path in env.paths], env._old_goals, filename='path.gif', with_lines=True)


# # %%

# # use matplotlib to visualize the map, where 0 is white and 1 is black
# plt.imshow(env.grid_map, cmap='gray_r')
# plt.show()