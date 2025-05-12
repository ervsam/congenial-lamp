import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from torchviz import make_dot


from utils import *
from Model import QNetwork, QJoint, VJoint

USE_GPU = False
if USE_GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

class Trainer:
    def __init__(self, LR, BATCH_SIZE, is_QTRAN_alt, LAMBDA, env):
        self.DEBUG = 0

        self.hidden_dim = 32

        self.env = copy.deepcopy(env)

        self.q_net = QNetwork().to(device)
        self.qjoint_net = QJoint().to(device)
        self.fixed_qjoint_net = QJoint().to(device)
        self.vnet = VJoint().to(device)

        self.update_every = 10
        self.counter = 0

        self.lr = LR
        self.batch_size = BATCH_SIZE
        self.num_agents = env.num_agents
        self.fov = env.fov
        self.is_QTRAN_alt = is_QTRAN_alt
        self.LAMBDA = LAMBDA

        # copy weights from q_net to fixed_qjoint_net
        self.fixed_qjoint_net.load_state_dict(self.qjoint_net.state_dict())
        self.fixed_qjoint_net.eval()
        for param in self.fixed_qjoint_net.parameters():
            param.requires_grad = False

        self.params = list({p for net in [self.q_net, self.qjoint_net, self.vnet] for p in net.parameters()})

        # self.optimizer = optim.Adam(self.params, lr=LR)
        self.optimizer = optim.AdamW(self.params, lr=LR, weight_decay=1e-4)

        self.criterion = nn.MSELoss(reduction="mean")

    def optimize(self, batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals, logger, weights):

        # move to device
        batch_obs = [obs.to(device) for obs in batch_obs]
        batch_global_reward = batch_global_reward.to(device)
        
        layers = 7
        # assert batch_obs.shape == (self.batch_size, self.num_agents, layers, self.fov, self.fov), f"Expected {(self.batch_size, self.num_agents, layers, self.fov, self.fov)}, got {batch_obs.shape}"

        num_n_pairs = []
        for pp in batch_partial_prio:
            num_n_pairs.append(len(pp.values()))

        batch_close_pairs = [list(pp.keys()) for pp in batch_partial_prio]
        if self.DEBUG:
            logger.print("batch_close_pairs:", batch_close_pairs)

        ### COMPUTE Q VALUES FOR EACH PAIR
        batch_pair_enc, batch_q_vals = self.q_net(batch_obs, batch_close_pairs)
        # batch_pair_enc: b x 'n_pairs' x h*2
        # batch_q_vals: b x 'n_pairs' x 2
        if self.DEBUG:
            logger.print("batch_pair_enc:", batch_pair_enc)

        # Vjt
        group_pair_enc = []
        for groups, pair_enc, close_pairs in zip(batch_groups, batch_pair_enc, batch_close_pairs):
            for group in groups:
                subgroup_pair_enc = []
                for pair, enc in zip(close_pairs, pair_enc):
                    if pair in group:
                        subgroup_pair_enc.append(enc)
            
                group_pair_enc.append(torch.stack(subgroup_pair_enc))
        
        # batch_pair_enc : 'groups' x 'n_pairs' x h*2
        vtot = torch.stack(self.vnet(group_pair_enc))
        # vtot = torch.zeros_like(q_jt)

        ### APPEND ACTIONS TO PAIR ENCODINGS
        batch_actions = []
        for pp in batch_partial_prio:
            batch_actions += list(pp.values())
        batch_actions = torch.stack(batch_actions).to(device)

        # batch_actions = torch.tensor([list(pp.values()) for pp in batch_partial_prio]).view(-1, 1)
        batch_onehot_actions = F.one_hot(batch_actions, num_classes=2).squeeze().to(device)
        if self.DEBUG:
            logger.print("batch_onehot_actions:", batch_onehot_actions)

        # flatten batch_q_vals
        tmp = []
        for q_vals in batch_q_vals:
            tmp += q_vals
        flatten_batch_q_vals = torch.stack(tmp)

        # flatten batch_pair_enc
        tmp = []
        for pair_enc in batch_pair_enc:
            tmp += pair_enc
        batch_pair_enc = torch.stack(tmp)

        batch_q_chosen = torch.gather(flatten_batch_q_vals, dim=1, index=batch_actions).squeeze()
        batch_pair_enc_action = torch.concat([batch_pair_enc, batch_onehot_actions], dim=1)
        if self.DEBUG:
            logger.print("batch_pair_enc_action:", batch_pair_enc_action)
            logger.print("shape of batch_pair_enc_action:", batch_pair_enc_action.shape)

        # Qjt
        q_jt, q_jt_alt = self.qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)
        # assert q_jt.shape == (BATCH_SIZE, 1), f"Expected {(BATCH_SIZE, 1)}, got {q_jt.shape}"
        fixed_q_jt, fixed_q_jt_alt = self.fixed_qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)
        # assert fixed_q_jt.shape == (BATCH_SIZE, 1), f"Expected {(BATCH_SIZE, 1)}, got {fixed_q_jt.shape}"

        # torchviz
        # dot = make_dot(q_jt_alt[0][0],  # ← be sure it’s a scalar
        #        params=dict(self.qjoint_net.named_parameters()))
        # dot.render('qtran_joint_graph', format='png')

        # keep only φ2 parameters
        # pars = dict(self.qjoint_net.phi2.named_parameters())

        # dot  = make_dot(q_jt_alt[0][0], params=pars,
        #                 show_attrs=False, show_saved=False)
        # dot.render('phi2_subgraph', format='png')


        ### GET MAX Q VALUES
        batch_max_actions = torch.argmax(flatten_batch_q_vals, dim=1).view(-1, 1)
        batch_onehot_max_actions = F.one_hot(batch_max_actions, num_classes=2).squeeze()
        batch_max_q = torch.gather(flatten_batch_q_vals, dim=1, index=batch_max_actions).squeeze()
        batch_pair_enc_max_action = torch.concat([batch_pair_enc, batch_onehot_max_actions], dim=1)

        # batch_pair_enc_max_action: b x 'n_pairs' x h*2+2
        q_jt_max, q_jt_max_alt = self.fixed_qjoint_net(batch_pair_enc, batch_pair_enc_max_action, batch_close_pairs, batch_groups, num_n_pairs)
        # assert q_jt_max.shape == (BATCH_SIZE, 1), f"Expected {(BATCH_SIZE, 1)}, got {q_jt_max.shape}"

        # Q'jt
        q_prime = []
        q_prime_max = []
        start = 0
        for num, close_pairs, groups in zip(num_n_pairs, batch_close_pairs, batch_groups):
            qvals = batch_q_chosen[start:start+num]
            group_qvals = []
            for group in groups:
                tmp_qs = []
                for q, pair in zip(qvals, close_pairs):
                    if pair in group:
                        tmp_qs.append(q)
                q_prime.append(sum(tmp_qs))
            # q_prime.append(group_qvals)

            max_qvals = batch_max_q[start:start+num]
            for group in groups:
                tmp_maxqs = []
                for max_q, pair in zip(max_qvals, close_pairs):
                    if pair in group:
                        tmp_maxqs.append(max_q)
                q_prime_max.append(sum(tmp_maxqs))

            start += num

        
        q_prime = torch.stack(q_prime).unsqueeze(1)
        q_prime_max = torch.stack(q_prime_max).unsqueeze(1)

        # CALCULATE LOSSES
        if self.is_QTRAN_alt:
            num_pairs_in_group = [] # len: batch_size * 'n_groups'
            for groups in batch_groups:
                for group in groups:
                    num_pairs_in_group.append(len(group))
            
            # q_jt_alt: b x 'n_pairs' x N_ACTIONS
            # flatten q_jt_alt
            tmp = []
            for t in q_jt_alt:
                tmp += t
            q_jt_alt = torch.stack(tmp) # b * 'n_pairs' x N_ACTIONS
            if self.DEBUG:
                logger.print("q_jt_alt:", q_jt_alt)

            tmp = []
            for t in fixed_q_jt_alt:
                tmp += t
            fixed_q_jt_alt = torch.stack(tmp) # b * 'n_pairs' x N_ACTIONS
            
            # print("q_jt_alt.shape, fixed_q_jt_alt.shape)", q_jt_alt.shape, fixed_q_jt_alt.shape)

            selected_Q = torch.sum(q_jt_alt * batch_onehot_actions, dim=1).unsqueeze(1) # b x 1

            if self.DEBUG:
                logger.print("selected_Q")
                logger.print(selected_Q)

            tmp = []
            for idx, num in enumerate(num_n_pairs):
                tmp += [batch_global_reward[idx]] * num
            batch_global_reward = torch.stack(tmp)

            tmp = []
            for r in batch_local_rewards:
                tmp += r
            batch_local_rewards = torch.tensor(tmp, dtype=torch.float32, device=device)

            # expand
            tmp = []
            for idx, num in enumerate(num_pairs_in_group):
                tmp += [batch_local_rewards[idx]] * num
            batch_local_rewards = torch.stack(tmp)

            # print("selected_Q.shape, batch_local_rewards.shape)", selected_Q.shape, batch_local_rewards.shape)
            assert selected_Q.shape == batch_local_rewards.unsqueeze(1).shape, f"Expected {selected_Q.shape}, got {batch_local_rewards.unsqueeze(1).shape}"

            ltd_raw = F.mse_loss(selected_Q, batch_local_rewards.unsqueeze(1), reduction='none').squeeze(1) 
            if self.DEBUG:
                logger.print("ltd_raw:")
                logger.print(ltd_raw)

            dup_factor = torch.tensor(num_pairs_in_group, device=device)
            norm = torch.tensor(dup_factor).repeat_interleave(dup_factor)
            if self.DEBUG:
                logger.print("norm:")
                logger.print(norm)
            ltd_pairs = ltd_raw / norm          # both (N,)

            if self.DEBUG:
                logger.print("ltd_pairs:")
                logger.print(ltd_pairs)

            # --- aggregate to group level -------------------------------------------
            group_ltd = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_ltd.append(ltd_pairs[cursor:cursor+n].sum())     # tensor with grad
                cursor += n
            group_ltd = torch.stack(group_ltd).to(device)                         # (G,) keeps grad

            # print group ltd
            if self.DEBUG:
                logger.print("ltd in group:", group_ltd)
                logger.print("group_ltd shape:", group_ltd.shape)
                         
            weights = torch.as_tensor(weights, device=device, dtype=group_ltd.dtype)
            exp_w = torch.repeat_interleave(weights, torch.as_tensor([len(g) for g in batch_groups], device=device))
            
            # --- aggregate to episode level -------------------------------------------
            # episode_ltd = []                                           # length = batch_size
            # cursor = 0
            # if self.DEBUG:
            #     logger.print("batch_groups len:", [len(groups) for groups in batch_groups])
            # for groups in batch_groups:                                # list-of-lists
            #     k = len(groups)
            #     episode_ltd.append(group_ltd[cursor:cursor+k].mean())  # tensor with grad
            #     cursor += k
            # episode_ltd = torch.stack(episode_ltd)                     # (B,) keeps grad

            # if self.DEBUG:
            #     logger.print("ltd in episode:", episode_ltd)

            # assert len(episode_ltd) == len(batch_obs), f"Expected {len(batch_obs)}, got {len(episode_ltd)}"

            # weights = torch.as_tensor(weights,                               # shape (B,)
            #         device=episode_ltd.device,
            #         dtype=episode_ltd.dtype)
            
            # if self.DEBUG:
            #     logger.print("weights:", weights)

            # y_is_rare = (batch_local_rewards.abs() > 0.5).float()      # 0 or 1 target

            # logit   = self.qjoint_net.rare_head(batch_pair_enc).squeeze()   # (total_pairs,)
            # bce     = F.binary_cross_entropy_with_logits(logit, y_is_rare)
            
            # ltd = (weights * episode_ltd).mean()        # scalar; gradient flows into Q-net
            ltd = (exp_w * group_ltd).mean()

            # ltd   = ltd + 0.2 * bce

            if self.DEBUG:
                logger.print("ltd:", ltd)

            tmp = torch.tensor([], device=device)
            for t in q_jt_max_alt:
                tmp = torch.concat([tmp, t])
            q_jt_max_alt = tmp

            max_Q = torch.sum(q_jt_max_alt * batch_onehot_max_actions, dim=1).unsqueeze(1)
            max_Q = max_Q.detach()

            # expand
            tmp = []
            for idx, num in enumerate(num_pairs_in_group):
                tmp += [q_prime_max[idx]] * num
            q_prime_max = torch.stack(tmp).to(device)
            tmp = []
            for idx, num in enumerate(num_pairs_in_group):
                tmp += [vtot[idx]] * num
            vtot = torch.stack(tmp).to(device)

            if self.DEBUG:
                print("q_prime_max.shape, max_Q.shape, vtot.shape)", q_prime_max.shape, max_Q.shape, vtot.shape)

            lopt_pairs = (q_prime_max - max_Q + vtot) ** 2
            # --- aggregate to group level -------------------------------------------
            group_lopt = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_lopt.append(lopt_pairs[cursor:cursor+n].sum())     # tensor with grad
                cursor += n
            group_lopt = torch.stack(group_lopt).to(device)                         # (G,) keeps grad

            if self.DEBUG:
                logger.print("group lopt:", group_lopt)
                logger.print("group lopt shape:", group_lopt.shape)

            lopt = group_lopt.mean()

            if self.DEBUG:
                logger.print("lopt in group:", lopt)

            # revert
            tmp = []
            start = 0
            for num in num_n_pairs:
                Q = flatten_batch_q_vals[start:start+num]
                tmp.append(Q)
                start += num
            q_uj = tmp # b x 'n_pairs' x N_ACTIONS

            q_is = flatten_batch_q_vals.gather(dim=1, index=batch_actions)
            tmp = []
            start = 0
            for num in num_n_pairs:
                q_i = q_is[start:start+num]
                tmp.append(q_i)
                start += num
            q_is = tmp # b x 'n_pairs' x 1

            q_prime_alt = []
            for uj, q_i in zip(q_uj, q_is):
                t = uj - q_i + torch.sum(q_i, dim=0, keepdim=True)
                q_prime_alt += (t)
            q_prime_alt = torch.stack(q_prime_alt).to(device)

            if self.DEBUG:
                print("q_prime_alt.shape, fixed_q_jt_alt.shape, vtot.shape)", q_prime_alt.shape, fixed_q_jt_alt.shape, vtot.shape)

            ########################################### Qjt(ui, u_-i) ############################################

            lnopt_min_pairs = torch.min(q_prime_alt - fixed_q_jt_alt.detach() + vtot, dim=1).values ** 2
            # --- aggregate to group level -------------------------------------------
            group_lnopt = []                                             # store tensors
            cursor = 0
            for n in num_pairs_in_group:                               # e.g. [2,8,5 …]
                group_lnopt.append(lnopt_min_pairs[cursor:cursor+n].sum())     # tensor with grad
                cursor += n
            group_lnopt = torch.stack(group_lnopt).to(device) # (G,) keeps grad

            if self.DEBUG:
                logger.print("group lnopt:", group_lnopt)
                logger.print("group lnopt shape:", group_lnopt.shape)

            lnopt_min = group_lnopt.mean()
            lnopt = lnopt_min

            loss = ltd + self.LAMBDA * lopt + self.LAMBDA * lnopt_min

            td_errors = []                 # length = batch_size
            cursor = 0
            for groups in batch_groups:      # list-of-lists
                k = len(groups)
                err = torch.max(group_ltd[cursor:cursor+k])
                td_errors.append(err)
                cursor += k
            td_errors = torch.stack(td_errors).to(device)
            td_errors = td_errors.detach().cpu().numpy()

            if self.DEBUG:
                logger.print("td_errors:", td_errors)

            tmp = []
            for idx, num in enumerate(num_pairs_in_group):
                tmp += [q_prime[idx]] * num
            q_prime = torch.stack(tmp).to(device)

        # not QTRAN_alt
        # else:
        #     # q_jt : b x 'groups' x 1
        #     tmp = []
        #     for q in q_jt:
        #         tmp += q
        #     q_jt = torch.stack(tmp)
            
        #     tmp = []
        #     for r in batch_local_rewards:
        #         tmp += r
        #     batch_local_rewards = torch.tensor(tmp, dtype=torch.float32)

        #     ltd = self.criterion(q_jt, batch_local_rewards.unsqueeze(1))

        #     # q_jt_max : b x 'groups' x 1
        #     tmp = []
        #     for q in q_jt_max:
        #         tmp += q
        #     q_jt_max = torch.stack(tmp)

        #     # q_prime_max : b x 'groups' x 1
        #     tmp = []
        #     for q in q_prime_max:
        #         tmp += q
        #     q_prime_max = torch.stack(tmp).unsqueeze(1)
            
        #     lopt = torch.mean((q_prime_max - q_jt_max + vtot) ** 2)

        #     # fixed_q_jt : b x 'groups' x 1
        #     tmp = []
        #     for q in fixed_q_jt:
        #         tmp += q
        #     fixed_q_jt = torch.stack(tmp)

        #     # q_prime : b x 'groups' x 1
        #     tmp = []
        #     for q in q_prime:
        #         tmp += q
        #     q_prime = torch.stack(tmp).unsqueeze(1)

        #     lnopt = torch.mean(torch.min(q_prime - fixed_q_jt + vtot, torch.zeros_like(fixed_q_jt)) ** 2)

        #     loss = ltd + self.LAMBDA * lopt + self.LAMBDA * lnopt
        #     # loss = self.LAMBDA * lopt + self.LAMBDA * lnopt
        #     # + torch.mean(vtot ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        logger.print("grad:", torch.nn.utils.clip_grad_norm_(self.qjoint_net.parameters(), max_norm=10))
        self.optimizer.step()

        self.counter += 1
        # update fixed_qjoint_net
        if self.counter % self.update_every == 0:
            self.fixed_qjoint_net.load_state_dict(self.qjoint_net.state_dict())

        return loss.item(), ltd.item(), lopt.item(), lnopt.item(), td_errors