import torch
import torch.nn as nn
import torch.nn.functional as F

# %% Encoder

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = F.leaky_relu(x)
        x = self.block2(x)
        x += identity
        x = F.leaky_relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, hid_dim=32):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.layers = 7

        # self.fc1 = nn.Linear(7 * 7 * 7, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        # self.fc_out = nn.Linear(64, 32)
        # self.bn1 = nn.BatchNorm1d(64)

        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        

        # self.Lin1 = nn.Linear(7 * 7 * 7, 256)
        # self.Lin2 = nn.Linear(256, 64)
        # self.Lin3 = nn.Linear(64, 32)
        # self.Lin4 = nn.Linear(32, 32)

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.layers, 32, 3, 1),
            nn.LeakyReLU(),
            ResBlock(32),
            ResBlock(32),

            # extra
            # ResBlock(32),
            # ResBlock(32),
            
            nn.Conv2d(32, 16, 1, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(400, self.hid_dim),
            # nn.Linear(128, 64),
            # nn.Linear(64, self.hid_dim)
        )

    def forward(self, x):
        # x: n x 7 x 7 x 7

        # x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(self.conv2(x))

        # # flatten
        # x = x.view(x.shape[0], -1)
        # x = F.leaky_relu(self.fc1(x))
        # # x = self.bn1(x)
        # x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        # x = F.leaky_relu(self.fc_out(x))


        # OPTION 2: FLATTEN AND LINEAR
        # x = x.view(x.size(0), -1)
        # x = F.leaky_relu(self.Lin1(x))
        # x = F.leaky_relu(self.Lin2(x))
        # x = F.leaky_relu(self.Lin3(x))
        # x = F.leaky_relu(self.Lin4(x))

        # OPTION 3: SACHA encoder
        x = self.obs_encoder(x)

        return x

# %% Q-Network (agent utility network)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.hid_dim = 32
        self.CONCAT = True

        self.encoder = Encoder(hid_dim=self.hid_dim)
        self.qnet = nn.Sequential(
            nn.Linear(self.hid_dim*2, self.hid_dim),
            nn.LeakyReLU(),
            # nn.Linear(self.hid_dim, self.hid_dim),
            # nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.hid_dim, num_heads=4, batch_first=True)
        self.lin1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin3 = nn.Linear(self.hid_dim, self.hid_dim)
        self.output_layer = nn.Linear(self.hid_dim, 1)

    def forward(self, batch_obs, batch_close_pairs) -> tuple[list, list]:
        '''
        AGENT UTILITY NETWORK

        Params
        ======
            batch_obs (torch tensor): batch of field of views
            batch_close_pairs (list): batch of list of close pairs of agents
        Returns
        =======
            n_pair_enc (list): batch of list of encoded pairs of agents
            batch_qvals (list): batch of list of Q values for each pair of agents
        '''

        # FOV = 3 x fov x fov

        # batch_obs: b x n x [FOV]
        batch_size = len(batch_obs)
        num_agents_list = [obs.shape[0] for obs in batch_obs]
        num_agents, channel, fov, _ = batch_obs[0].shape

        # b x n x [FOV] -> b*n x [FOV]
        tmp = []
        for b in batch_obs:
            for n in b:
                tmp.append(n)
        batch_obs = torch.stack(tmp)
        # batch_obs = batch_obs.view(-1, channel, fov, fov)

        ##############################################
        # ENCODE
        # reshape b*n x [FOV] -> b*n x h
        batch_encoded_obs = self.encoder(batch_obs)

        # assert batch_encoded_obs.shape == (batch_size * num_agents, self.hid_dim), \
        #     f"Expected {(batch_size * num_agents, self.hid_dim)}, got {batch_encoded_obs.shape}"

        # reshape b*n x h -> b x n x h
        # batch_encoded_obs = batch_encoded_obs.view(batch_size, num_agents, -1)
        tmp = []
        start = 0
        for num_agents in num_agents_list:
            t = batch_encoded_obs[start:start+num_agents]
            tmp.append(t)
            start += num_agents
        batch_encoded_obs = tmp

        # assert batch_encoded_obs.shape == (batch_size, num_agents, self.hid_dim), \
        #     f"Expected {(batch_size, num_agents, self.hid_dim)}, got {batch_encoded_obs.shape}"

        ##############################################
        # CALCULATE AGENT UTILITY

        batch_qvals = []  # b x 'n_pairs' x 2
        n_pair_enc = []

        # for each data point in the batch, calculate Q values for each pair of agents
        for close_pairs, encoded_obs in zip(batch_close_pairs, batch_encoded_obs):
            concat_pair_encodings = []  # 'n_pairs' x h*2
            stack_pair_encodings = []  # 'n_pairs' x 2 x h

            # for each close pair, calculate Q values
            for agents in close_pairs:
                agent_1, agent_2 = agents[0], agents[1]
                concat_pair_obs = torch.concatenate((encoded_obs[agent_1], encoded_obs[agent_2]))

                # 2 x h
                stack_pair_obs = torch.stack(([encoded_obs[agent_1], encoded_obs[agent_2]]))
                assert stack_pair_obs.shape == (2, self.hid_dim), \
                    f"Expected {(2, self.hid_dim)}, got {stack_pair_obs.shape}"

                concat_pair_encodings.append(concat_pair_obs)
                stack_pair_encodings.append(stack_pair_obs)

            concat_pair_encodings = torch.stack(concat_pair_encodings)
            stack_pair_encodings = torch.stack(stack_pair_encodings)

            # OPTION 1: CONCATENATION
            if self.CONCAT:
                hid3 = concat_pair_encodings
                qvals = self.qnet(concat_pair_encodings)
                batch_qvals.append(qvals)

            # OPTION 2: ATTENTION
            else:
                assert stack_pair_encodings.shape == (
                    len(close_pairs), 2, self.hid_dim), f"Expected {(len(close_pairs), 2, self.hid_dim)}, got {stack_pair_encodings.shape}"
                
                # print("stack_pair_encodings")
                # print(stack_pair_encodings)

                obs_attended, attn_output_weights = self.attention_layer(stack_pair_encodings, stack_pair_encodings, stack_pair_encodings)
                
                # print("obs_attended")
                # print(obs_attended)
                
                output = F.leaky_relu(self.lin1(obs_attended))
                output = F.leaky_relu(self.lin2(output))
                hid3 = F.leaky_relu(self.lin3(output))

                output = self.output_layer(hid3)

                batch_qvals.append(output.squeeze())

            # print("hid3")
            # print(hid3) # 'n_pairs' x 2 x h
            n_pair_enc.append(hid3.view(hid3.shape[0], -1))
            # n_pair_enc.append(concat_pair_encodings)

        # assert shape of batch_qvals is (batch_size, 'n_pairs', 2)
        assert len(batch_qvals) == batch_size, f"Expected {batch_size}, got {len(batch_qvals)}"
        assert batch_qvals[0].shape[1] == 2, f"Expected {2}, got {batch_qvals[0].shape[1]}"

        # b x n_pairs x h*2, b x 'n_pairs' x 2
        return n_pair_enc, batch_qvals


# %% Joint Q-Network

class QJoint(nn.Module):
    def __init__(self):
        super(QJoint, self).__init__()

        self.input_dim = 66
        self.hidden_dim = 64

        self.linear_0 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.ELU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ELU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.lin1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, 2)


    def forward(self, batch_pair_enc, batch_pair_enc_action, batch_close_pairs, batch_groups, num_n_pairs):

        # expand
        tmp = []
        tmp_batch_pair_enc = []
        start = 0
        for num in num_n_pairs:
            t = batch_pair_enc_action[start:start+num]
            tmp.append(t)

            t = batch_pair_enc[start:start+num]
            tmp_batch_pair_enc.append(t)

            start += num
        batch_pair_enc_action = tmp
        key2 = tmp_batch_pair_enc

        # batch_pair_enc_action: b x 'n_pairs' x h*2+2

        batch_size = len(batch_pair_enc_action)
        b_qjt = []
        joint_qs_prereduced = []

        for pair_enc_action, close_pairs, groups in zip(batch_pair_enc_action, batch_close_pairs, batch_groups):
            # pair_enc_action: 'n_pairs' x h*2+2
            assert pair_enc_action.shape[1] == self.input_dim, f"Expected {self.input_dim}, got {pair_enc_action.shape[1]}"

            key1 = self.linear_0(pair_enc_action) # 'n_pairs' x h

            # print("groups:", groups)
            grouped_key1 = []
            for group in groups:
                tmp_grp = []
                for k1, pair in zip(key1, close_pairs):
                    if pair in group:
                        tmp_grp.append(k1)
                grouped_key1.append(torch.stack(tmp_grp))
            joint_qs_prereduced += grouped_key1


            group_jt = []
            for k1 in grouped_key1:
                # mean encoded q_vals
                combined_agents = k1.mean(dim=0) # h
                q_jt = self.linear(combined_agents)

                assert q_jt.shape == (1,), f"Expected {(1,)}, got {q_jt.shape}"
                group_jt.append(q_jt)
            
            b_qjt.append(torch.stack(group_jt))

        # b_qjt: b x 'group' x 1
        # assert b_qjt.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {b_qjt.shape}"


        # joint_qs_prereduced: b*'group' x 'n_pairs' x h
        averaged = []
        for q in joint_qs_prereduced:
            # 'n_pairs' x h
            averaged.append(q.mean(dim=0, keepdim=True))
        averaged = torch.stack(averaged) # 'group' x 1 x h

        grouped_key2 = []
        for k2s, close_pairs, groups in zip(key2, batch_close_pairs, batch_groups):
            # k2: 'n_pairs' x h
            grouped_k2s = []
            for group in groups:
                tmp_grp = []
                for k2, pair in zip(k2s, close_pairs):
                    if pair in group:
                        tmp_grp.append(k2)
                grouped_k2s.append(torch.stack(tmp_grp))
            grouped_key2 += grouped_k2s

        # grouped_key2: 'group' x 'n_pairs' x h
        # averaged: 'group' x 1 x h
        # joint_qs_prereduced: 'group' x 'n_pairs' x h
        b_qjt_alt = []
        for k2, ave, j in zip(grouped_key2, averaged, joint_qs_prereduced):
            tmp = k2 + ave - j/len(j) # len(j) = 'n_pairs'
            b_qjt_alt.append(tmp)
        # b_qjt_alt = key2 + averaged - joint_qs_prereduced/'n_pairs'
        tmp = []
        for q in b_qjt_alt:
            q = F.elu(self.lin1(q))
            q = self.lin2(q)
            tmp.append(q)
        b_qjt_alt = tmp

        return b_qjt, b_qjt_alt

# %% Joint V Network

class VJoint(nn.Module):
    def __init__(self):
        super(VJoint, self).__init__()

        self.v_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, n_pair_enc):

        # n_pair_enc: b x 'n_pairs' x h*2

        vtot = []
        for pair_enc in n_pair_enc:
            # 'n_pairs' x h
            v = self.v_net(pair_enc.sum(dim=0))
            # 1 x h
            vtot.append(v)
        # b x 1 x h

        return vtot
