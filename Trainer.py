import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from utils import *
from Model import QNetwork, QJoint, VJoint

class Trainer:
    def __init__(self, LR, BATCH_SIZE, NUM_AGENTS, FOV, is_QTRAN_alt, LAMBDA, env):
        self.hidden_dim = 32

        self.env = copy.deepcopy(env)

        self.q_net = QNetwork()
        self.qjoint_net = QJoint()
        # self.fixed_qjoint_net = QJoint()
        self.vnet = VJoint()

        self.update_every = 100
        self.counter = 0

        self.lr = LR
        self.batch_size = BATCH_SIZE
        self.num_agents = NUM_AGENTS
        self.fov = FOV
        self.is_QTRAN_alt = is_QTRAN_alt
        self.LAMBDA = LAMBDA

        # copy weights from q_net to fixed_qjoint_net
        # self.fixed_qjoint_net.load_state_dict(self.qjoint_net.state_dict())
        # self.fixed_qjoint_net.eval()
        # for param in self.fixed_qjoint_net.parameters():
        #     param.requires_grad = False

        # self.params = list(self.q_net.parameters()) + list(self.qjoint_net.parameters()) + list(self.vnet.parameters())
        self.params = list({p for net in [self.q_net, self.vnet] for p in net.parameters()})
        self.optimizer = optim.Adam(self.params, lr=LR)
        self.criterion = nn.MSELoss()

    def optimize(self, batch_obs, batch_partial_prio, batch_global_reward, batch_local_rewards, batch_groups, batch_starts, batch_goals):
        layers = 7
        assert batch_obs.shape == (self.batch_size, self.num_agents, layers, self.fov, self.fov), f"Expected {(self.batch_size, self.num_agents, layers, self.fov, self.fov)}, got {batch_obs.shape}"

        num_n_pairs = []
        for pp in batch_partial_prio:
            num_n_pairs.append(len(pp.values()))

        batch_close_pairs = [list(pp.keys()) for pp in batch_partial_prio]

        ### COMPUTE Q VALUES FOR EACH PAIR
        batch_pair_enc, batch_q_vals = self.q_net(batch_obs, batch_close_pairs)
        # batch_pair_enc: b x 'n_pairs' x h*2
        # batch_q_vals: b x 'n_pairs' x 2

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
        batch_actions = torch.stack(batch_actions)
        # batch_actions = torch.tensor([list(pp.values()) for pp in batch_partial_prio]).view(-1, 1)
        batch_onehot_actions = F.one_hot(batch_actions, num_classes=2).squeeze()

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

        # Qjt
        # q_jt, q_jt_alt = self.qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)
        # assert q_jt.shape == (BATCH_SIZE, 1), f"Expected {(BATCH_SIZE, 1)}, got {q_jt.shape}"
        # fixed_q_jt, fixed_q_jt_alt = self.fixed_qjoint_net(batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs)
        # assert fixed_q_jt.shape == (BATCH_SIZE, 1), f"Expected {(BATCH_SIZE, 1)}, got {fixed_q_jt.shape}"

        # batch_pair_enc_action = []
        # batch_q_chosen = []
        # # for each instance in batch
        # for partial_prio, q_vals, pair_enc in zip(batch_partial_prio, batch_q_vals, batch_pair_enc):
        #     # pair_enc: 'n_pairs' x h*2

        #     # one hot encoding of actions
        #     # and append to pair_enc
        #     actions = torch.stack(list(partial_prio.values())).squeeze()
        #     pair_enc_action = torch.concat([pair_enc, F.one_hot(actions, num_classes=2)], dim=1)
        #     assert pair_enc_action.shape == (len(partial_prio), 66), f"Expected {(len(partial_prio), 66)}, got {pair_enc_action.shape}"

        #     q_values = q_vals.gather(dim=1, index=actions)
        #     assert q_values.shape == (len(partial_prio), 1), f"Expected {(len(partial_prio), 1)}, got {q_values.shape}"
            
        #     batch_q_chosen.append(q_values)
        #     batch_pair_enc_action.append(pair_enc_action)
        #     # batch_pair_enc_action: b x 'n_pairs' x h*2+2

        ### GET MAX Q VALUES
        batch_max_actions = torch.argmax(flatten_batch_q_vals, dim=1).view(-1, 1)
        batch_onehot_max_actions = F.one_hot(batch_max_actions, num_classes=2).squeeze()
        batch_max_q = torch.gather(flatten_batch_q_vals, dim=1, index=batch_max_actions).squeeze()
        batch_pair_enc_max_action = torch.concat([batch_pair_enc, batch_onehot_max_actions], dim=1)

        # batch_pair_enc_max_action: b x 'n_pairs' x h*2+2
        # q_jt_max, q_jt_max_alt = self.fixed_qjoint_net(batch_pair_enc, batch_pair_enc_max_action, batch_close_pairs, batch_groups, num_n_pairs)
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
                group_qvals.append(sum(tmp_qs))
            q_prime.append(group_qvals)

            max_qvals = batch_max_q[start:start+num]
            for group in groups:
                tmp_maxqs = []
                for max_q, pair in zip(max_qvals, close_pairs):
                    if pair in group:
                        tmp_maxqs.append(max_q)
                q_prime_max.append(sum(tmp_maxqs))

            start += num

        
        # q_prime = torch.stack(q_prime).unsqueeze(1)
        q_prime_max = torch.stack(q_prime_max).unsqueeze(1)

        # CALCULATE LOSSES
        if self.is_QTRAN_alt:
            # q_jt_alt: b x 'n_pairs' x N_ACTIONS
            # flatten q_jt_alt
            # tmp = []
            # for t in q_jt_alt:
            #     tmp += t
            # q_jt_alt = torch.stack(tmp) # b * 'n_pairs' x N_ACTIONS

            l = []
            for groups in batch_groups:
                for group in groups:
                    l.append(len(group))

            # tmp = []
            # for t in fixed_q_jt_alt:
            #     tmp += t
            # fixed_q_jt_alt = torch.stack(tmp) # b * 'n_pairs' x N_ACTIONS
            
            # print("q_jt_alt.shape, fixed_q_jt_alt.shape)", q_jt_alt.shape, fixed_q_jt_alt.shape)

            # selected_Q = torch.sum(q_jt_alt * batch_onehot_actions, dim=1).unsqueeze(1) # b x 1


            tmp = []
            for idx, num in enumerate(num_n_pairs):
                tmp += [batch_global_reward[idx]] * num
            batch_global_reward = torch.stack(tmp)

            tmp = []
            for r in batch_local_rewards:
                tmp += r
            batch_local_rewards = torch.tensor(tmp, dtype=torch.float32)

            # expand
            tmp = []
            for idx, num in enumerate(l):
                tmp += [batch_local_rewards[idx]] * num
            batch_local_rewards = torch.stack(tmp)

            # print("selected_Q.shape, batch_local_rewards.shape)", selected_Q.shape, batch_local_rewards.shape)


            # ltd = self.criterion(selected_Q, batch_local_rewards.unsqueeze(1))
            # ltd = torch.tensor(0)

            # tmp = torch.tensor([])
            # for t in q_jt_max_alt:
            #    tmp = torch.concat([tmp, t])
            # q_jt_max_alt = tmp



            # 1
            # batch_max_actions, batch_q_vals
            # priorities, partial_prio, pred_value = sample_priorities(env, logger, env.get_close_pairs(), q_vals[0], policy=policy)
            tmp = []
            start = 0
            for num in num_n_pairs:
                Q = batch_max_actions[start:start+num]
                tmp.append(Q)
                start += num
            batch_max_actions = tmp # b x 'n_pairs' x N_ACTIONS

            batch_max_local_rewards = []
            for starts, goals, close_pairs, q_vals, max_actions in zip(batch_starts, batch_goals, batch_close_pairs, batch_q_vals, batch_max_actions):
                self.env.starts = starts.copy()
                self.env.goals = goals.copy()

                pairs_action = {}
                pair_qval = {}
                edges = []
                for pair, confidence, action in zip(close_pairs, q_vals, max_actions):
                    t = round((max(confidence) - min(confidence)).item(), 3)
                    if action == 1:
                        pair_qval[pair[::-1]] = t
                        edges.append(pair[::-1])
                    else:
                        pair_qval[pair] = t
                        edges.append(pair)
                    pairs_action[pair] = action
                pair_qval = dict(sorted(pair_qval.items(), key=lambda item: item[1], reverse=True))

                graph = DirectedGraph(self.env.num_agents)
                for pair, (u, v) in zip(close_pairs, edges):
                    if graph.add_edge(u, v): # if edge doesn't form a cycle
                        continue
                    else:
                        pairs_action[pair] = 1 - pairs_action[pair]

                actions = list(pairs_action.values())

                partial_prio = dict()
                for i, (agent, neighbor) in enumerate(close_pairs):
                    partial_prio[(agent, neighbor)] = actions[i]

                priorities, cycle = topological_sort(partial_prio, num_nodes=self.env.num_agents).values()
                # 2
                new_start, new_goals = self.env.step(priorities)

                start_time = time.time()
                while new_start is None and time.time() - start_time < 60:
                    probs = F.softmax(q_vals, dim=1)
                    actions = torch.multinomial(probs, 1)
                    pairs_action = {}
                    pair_qval = {}
                    edges = []
                    for pair, confidence, action in zip(close_pairs, q_vals, actions):
                        t = round((max(confidence) - min(confidence)).item(), 3)
                        if action == 1:
                            pair_qval[pair[::-1]] = t
                            edges.append(pair[::-1])
                        else:
                            pair_qval[pair] = t
                            edges.append(pair)
                        pairs_action[pair] = action
                    pair_qval = dict(sorted(pair_qval.items(), key=lambda item: item[1], reverse=True))

                    graph = DirectedGraph(self.env.num_agents)
                    for pair, (u, v) in zip(close_pairs, edges):
                        if graph.add_edge(u, v): # if edge doesn't form a cycle
                            continue
                        else:
                            pairs_action[pair] = 1 - pairs_action[pair]
                    actions = list(pairs_action.values())

                    partial_prio = dict()
                    for i, (agent, neighbor) in enumerate(close_pairs):
                        partial_prio[(agent, neighbor)] = actions[i]

                    priorities, cycle = topological_sort(partial_prio, num_nodes=self.env.num_agents).values()
                    
                    new_start, new_goals = self.env.step(priorities)


                # 3
                local_rewards = []
                group_agents = []
                if new_start is None:
                    for group in group_agents:
                        local_rewards.append(-50)
                else:
                    delays = self.env.get_delays()
                    for group in groups:
                        set_s = set()
                        for pair in group:
                            set_s.update(pair)
                        group_agents.append(list(set_s))
                    for group in group_agents:
                        local_rewards.append(sum([-delays[agent] for agent in group]))

                batch_max_local_rewards += local_rewards

            # expand
            tmp = []
            for idx, num in enumerate(l):
                tmp += [batch_max_local_rewards[idx]] * num
            batch_max_local_rewards = torch.tensor(tmp, dtype=torch.float32)

            # max_Q = torch.sum(q_jt_max_alt * batch_onehot_max_actions, dim=1).unsqueeze(1)
            # max_Q = max_Q.detach()


            # expand
            tmp = []
            for idx, num in enumerate(l):
                tmp += [q_prime_max[idx]] * num
            q_prime_max = torch.stack(tmp)
            tmp = []
            for idx, num in enumerate(l):
                tmp += [vtot[idx]] * num
            vtot = torch.stack(tmp)

            # vtot = torch.zeros_like(vtot)

            # print("q_prime_max.shape, max_Q.shape, vtot.shape)", q_prime_max.shape, max_Q.shape, vtot.shape)

            # lopt = torch.mean((q_prime_max - max_Q + vtot) ** 2)
            # assert q_prime_max.shape == batch_max_local_rewards.unsqueeze(1).shape, f"Expected {q_prime_max.shape}, got {batch_max_local_rewards.unsqueeze(1).shape}"
            lopt = torch.mean((q_prime_max - batch_max_local_rewards + vtot) ** 2)

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
            q_prime_alt = torch.stack(q_prime_alt)

            # print("q_prime_alt.shape, fixed_q_jt_alt.shape, vtot.shape)", q_prime_alt.shape, fixed_q_jt_alt.shape, vtot.shape)

            # lnopt_min = torch.mean(torch.min(q_prime_alt - fixed_q_jt_alt.detach() + vtot, dim=1).values ** 2)
            # assert q_prime_alt.shape == batch_local_rewards.unsqueeze(1).shape, f"Expected {q_prime_alt.shape}, got {batch_local_rewards.unsqueeze(1).shape}"
            lnopt_min = torch.mean(torch.min(q_prime_alt - batch_local_rewards.unsqueeze(1) + vtot, dim=1).values ** 2)

            # loss = ltd + self.LAMBDA * lopt + self.LAMBDA * lnopt_min
            loss = self.LAMBDA * lopt + self.LAMBDA * lnopt_min
            # + torch.mean(vtot ** 2)

            lnopt = lnopt_min

            # a = torch.abs(selected_Q - batch_local_rewards.unsqueeze(1)).detach().numpy()
            b = torch.abs(batch_max_local_rewards - q_prime_max).detach().numpy()
            c = np.mean(torch.abs(q_prime_alt - batch_local_rewards.unsqueeze(1)).detach().numpy(), axis=1, keepdims=True)
            td_errors = b + c

            tmp = []
            start = 0
            for num in num_n_pairs:
                Q = np.mean(td_errors[start:start+num])
                tmp.append(Q)
                start += num
            td_errors = tmp # b x N_ACTIONS
        else:
            # q_jt : b x 'groups' x 1
            tmp = []
            for q in q_jt:
                tmp += q
            q_jt = torch.stack(tmp)
            
            tmp = []
            for r in batch_local_rewards:
                tmp += r
            batch_local_rewards = torch.tensor(tmp, dtype=torch.float32)

            ltd = self.criterion(q_jt, batch_local_rewards.unsqueeze(1))

            # q_jt_max : b x 'groups' x 1
            tmp = []
            for q in q_jt_max:
                tmp += q
            q_jt_max = torch.stack(tmp)

            # q_prime_max : b x 'groups' x 1
            tmp = []
            for q in q_prime_max:
                tmp += q
            q_prime_max = torch.stack(tmp).unsqueeze(1)
            
            lopt = torch.mean((q_prime_max - q_jt_max + vtot) ** 2)

            # fixed_q_jt : b x 'groups' x 1
            tmp = []
            for q in fixed_q_jt:
                tmp += q
            fixed_q_jt = torch.stack(tmp)

            # q_prime : b x 'groups' x 1
            tmp = []
            for q in q_prime:
                tmp += q
            q_prime = torch.stack(tmp).unsqueeze(1)

            lnopt = torch.mean(torch.min(q_prime - fixed_q_jt + vtot, torch.zeros_like(fixed_q_jt)) ** 2)

            # loss = ltd + self.LAMBDA * lopt + self.LAMBDA * lnopt
            loss = self.LAMBDA * lopt + self.LAMBDA * lnopt
            # + torch.mean(vtot ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update fixed_qjoint_net
        self.counter += 1
        if self.counter % self.update_every == 0:
            self.fixed_qjoint_net.load_state_dict(self.qjoint_net.state_dict())

        # return loss.item(), ltd.item(), lopt.item(), lnopt.item(), td_errors
        return loss.item(), 0, lopt.item(), lnopt.item(), td_errors