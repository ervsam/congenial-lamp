import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

LATENT_DIM = 64

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
    def __init__(self, hid_dim: int = LATENT_DIM):
        super().__init__()
        self.layers = 7          # image channels

        self.conv = nn.Sequential(
            nn.Conv2d(self.layers, 32, 3, padding=1), nn.LeakyReLU(),
            ResBlock(32),                             # <- keep two res-blocks
            ResBlock(32),
            nn.Dropout2d(p=0.10),                     # not 0.20
            nn.Conv2d(32, 16, 1),  nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16*11*11, 64),  nn.LeakyReLU(), # 64-d latent
        )
        self.coord_fc = nn.Linear(2, 32)
        self.out      = nn.Linear(64+32, 64)  

    def forward(self, x, coords):
        h_img   = self.conv(x)
        h_coord = self.coord_fc(coords)
        h       = torch.cat([h_img, h_coord], dim=-1)
        return self.out(h)
    
# %% Encoder (old version, not used)
# class Encoder(nn.Module):
#     def __init__(self, hid_dim=32):
#         super(Encoder, self).__init__()

#         self.hid_dim = hid_dim
#         self.layers = 7

#         self.obs_encoder = nn.Sequential(
#             nn.Conv2d(self.layers, 32, 3, 1),
#             nn.LeakyReLU(),
#             ResBlock(32),
#             ResBlock(32),

#             nn.Conv2d(32, 16, 1, 1),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             nn.Linear(1296, self.hid_dim),
#         )

#         self.encode_coords = nn.Linear(2, 32)
#         self.output = nn.Linear(self.hid_dim + 32, self.hid_dim)

#     def forward(self, x, coords):
#         # x: n x 7 x 7 x 7
#         x = self.obs_encoder(x)
#         c = self.encode_coords(coords)
#         x = torch.cat((x, c), dim=1)
#         x = self.output(x)

#         return x

# %% Q-Network (agent utility network)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.CONCAT = True

        self.hid_dim = LATENT_DIM
        self.encoder = Encoder(hid_dim=self.hid_dim)

        # utility net (concatenation path)
        self.qnet = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hid_dim, 2),        # 2 actions
        )

    def forward(self,
                batch_obs: list[torch.Tensor],
                batch_close_pairs: list[list[tuple[int,int]]]
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Parameters
        ----------
        batch_obs
            List of length B.  obs_i has shape (N_i, C, F, F).
            The **last** input channel is assumed to hold the (x,y) coordinates.
        batch_close_pairs
            Same length (B).  Each element is a list of (a,b) index pairs
            referring to the agents in that episode.

        Returns
        -------
        n_pair_enc    - list length B, each item is (n_pairs_i, 2*H)            # encoder outputs per pair
        batch_qvals   - list length B, each item is (n_pairs_i, 2)              # utility Q-values per pair
        """

        #### ---------------------------------------------------------------- ##
        #### 0.  Episode-level bookkeeping
        #### ---------------------------------------------------------------- ##
        device   = batch_obs[0].device
        batch_size = len(batch_obs)
        num_agents_list = [obs.shape[0] for obs in batch_obs]       # e.g. [4,6,5,…]

        #### ---------------------------------------------------------------- ##
        #### 1.  Flatten → encode every agent once
        #### ---------------------------------------------------------------- ##
        fov_flat, coord_flat = [], []
        for obs in batch_obs:
            fov_flat.append(obs[:, :-1])             # (N_i, C-1, F, F)
            coord_flat.append(obs[:, -1, 0, 0:2])    # (N_i, 2)
        batch_obs   = torch.cat(fov_flat,   0).to(device)    # (ΣN_i, C-1, F, F)
        batch_coordinates = torch.cat(coord_flat, 0).to(device)    # (ΣN_i, 2)

        flat_encoded_obs   = self.encoder(batch_obs, batch_coordinates)     # (ΣN_i, H)

        # split back – tuple of tensors, one per episode
        batch_enc = torch.split(flat_encoded_obs, num_agents_list, dim=0)   # B × (N_i, H)

        #### ---------------------------------------------------------------- ##
        #### 2.  Build pair encodings + utility head for each episode
        #### ---------------------------------------------------------------- ##
        pair_enc_per_ep : list[torch.Tensor] = []
        qvals_per_ep    : list[torch.Tensor] = []

        for enc_agents, pairs in zip(batch_enc, batch_close_pairs):
            if len(pairs) == 0:        # safety
                pair_enc_per_ep.append(torch.empty(0, self.hid_dim*2, device=device))
                qvals_per_ep.append(torch.empty(0, 2, device=device))
                continue

            # gather encoded agents
            idx_a = torch.tensor([p[0] for p in pairs], device=device)
            idx_b = torch.tensor([p[1] for p in pairs], device=device)

            h_a   = enc_agents[idx_a]               # (n_pairs, H)
            h_b   = enc_agents[idx_b]               # (n_pairs, H)

            if self.CONCAT:
                pair_enc = torch.cat([h_a, h_b], dim=-1)     # (n_pairs, 2H)
                pair_q   = self.qnet(pair_enc)               # (n_pairs, 2)

                pair_enc_per_ep.append(pair_enc)             # store
                qvals_per_ep   .append(pair_q)

        assert len(qvals_per_ep) == batch_size, f"Expected {batch_size}, got {len(qvals_per_ep)}"
        assert qvals_per_ep[0].shape[1] == 2, f"Expected {2}, got {qvals_per_ep[0].shape[1]}"
        return pair_enc_per_ep, qvals_per_ep


# %% Joint Q-Network
class QJoint(nn.Module):
    def __init__(self):
        super(QJoint, self).__init__()

        self.hid = LATENT_DIM
        self.in_dim = self.hid * 2 + 2        # encA ⊕ encB ⊕ one-hot

        # φ₁  (key1) – encodes each pair
        self.phi1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hid, self.hid * 2),
        )

        # g – combines the mean of φ₁ into a scalar Qjt
        self.g = nn.Sequential(
            nn.Linear(self.hid * 2, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hid, 1),
        )

        # φ₂  (key2) – value for alternative head
        self.phi2 = nn.Sequential(
            nn.Linear(self.hid * 2, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, 2),          # outputs per-pair Q-vector
        )

        # self.rare_head = nn.Linear(self.hid * 2, 1)    # in QJoint.__init__()


        # old version
        self.input_dim = LATENT_DIM * 2 + 2
        self.hidden_dim = LATENT_DIM
        # self.linear_0 = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ELU(),
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ELU(),
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ELU(),
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ELU(),
        #     nn.Linear(self.hidden_dim, 64),
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(64, self.hidden_dim),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim*2, self.hidden_dim),
        #     # nn.ELU(),
        #     # nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ELU(),
        #     nn.Linear(self.hidden_dim, 1),
        # )
        # self.lin1 = nn.Linear(64, 64)
        # self.lin2 = nn.Linear(64, 2)


    def forward(self,
                flat_pair_enc  : torch.Tensor,           # (ΣP, 2H)  – no actions
                flat_pair_encA : torch.Tensor,           # (ΣP, 2H+2)– enc ⊕ one-hot
                batch_pairs    : list[list[tuple[int,int]]],
                batch_groups   : list[list[list[tuple[int,int]]]],
                n_pairs_list   : list[int]               # [#pairs_epi_0, #pairs_epi_1, …]
            ):
    
        """
        Returns
        -------
        b_qjt       - list length B, each item shape (n_groups_i, 1)
        b_qjt_alt   - list length B, each item shape (n_groups_i, n_pairs_in_group, 2)
        """

        #### -------------------------------------------------------------- ##
        #### 0.  Split flat tensors back to episodes
        #### -------------------------------------------------------------- ##
        pairA_per_ep = torch.split(flat_pair_encA, n_pairs_list, dim=0)     # B × (P_i, 2H+2)
        key2_per_ep  = torch.split(flat_pair_enc,  n_pairs_list, dim=0)     # B × (P_i, 2H)

        out_qjt, out_qjt_alt = [], []

        # OLD (DELETE)
        # expand
        # tmp = []
        # tmp_batch_pair_enc = []
        # start = 0
        # for num in n_pairs_list:
        #     t = batch_pair_enc_action[start:start+num]
        #     tmp.append(t)

        #     t = batch_pair_enc[start:start+num]
        #     tmp_batch_pair_enc.append(t)

        #     start += num
        # batch_pair_enc_action = tmp
        # key2 = tmp_batch_pair_enc

        #### -------------------------------------------------------------- ##
        #### 1.  Per-episode computation (tiny loops; remain on GPU)
        #### -------------------------------------------------------------- ##
        for pairA, key2, pairs, groups in zip(pairA_per_ep,
                                            key2_per_ep,
                                            batch_pairs,
                                            batch_groups):
            
            # ------------------------------------------------------------------ #
            # 1-a  encode every pair once  (φ₁)
            # ------------------------------------------------------------------ #
            key1 = self.phi1(pairA)                     # (P_i, H)

            # quick map:   pair-tuple  → index 0…P_i-1
            pair2idx = {p: idx for idx, p in enumerate(pairs)}

            # ------------------------------------------------------------------ #
            # 1-b  group-wise mean of φ₁  (Eq. 8 – pre-reduce)
            # ------------------------------------------------------------------ #
            # Build a flat index list that says for each pair which group it
            # belongs to; then scatter_mean does the averaging on GPU.
            group_index = torch.empty(len(pairs), dtype=torch.long, device=key1.device)
            for g_id, g in enumerate(groups):
                for p in g:
                    group_index[pair2idx[p]] = g_id
            # key1_mean[g] = mean_{p∈g} key1[p]
            key1_mean = scatter_mean(key1, group_index, dim=0)              # (G_i, H)

            # scalar Qjt per group   (Eq. 9)
            q_jt = self.g(key1_mean).squeeze(-1)  

            
            # ------------------------------------------------------------------ #
            # 1-c  Alternative head  (Eq. 10   k₂ + k̄₁ − k₁/|g|)
            # ------------------------------------------------------------------ #
            # Need broadcasted tensors:
            #   k₂[p],   mean_key1_of_group[p],   key1[p]/|g|
            #
            k2            = key2                                             # (P_i, 2H)
            k1_div_len_g  = key1 / scatter_mean(torch.ones_like(key1),
                                                group_index, dim=0)[group_index]  # (P_i,H)

            kbar1         = key1_mean[group_index]                           # (P_i, H)
            alt_val       = k2 + kbar1 - k1_div_len_g                        # (P_i,2H)  (2H if φ₂ expects that)
            alt_q         = self.phi2(alt_val)                               # (P_i, 2)

            # reorganise alt_q into (G_i, n_pairs_in_g, 2) for convenience
            splits   = [len(g) for g in groups]
            alt_per_g = alt_q.split(splits, dim=0)
            out_qjt.append(q_jt.unsqueeze(-1))        # keep shape (G_i,1) to match old code
            out_qjt_alt += alt_per_g

        return out_qjt, out_qjt_alt 


        # # OLD (DELETE)
        # joint_qs_prereduced = []

        # for pair_enc_action, close_pairs, groups in zip(batch_pair_enc_action, batch_close_pairs, batch_groups):
        #     assert pair_enc_action.shape[1] == self.input_dim, f"Expected {self.input_dim}, got {pair_enc_action.shape[1]}"

        #     key1 = self.phi1(pair_enc_action)

        #     # print("groups:", groups)
        #     grouped_key1 = []
        #     for group in groups:
        #         tmp_grp = []
        #         for k1, pair in zip(key1, close_pairs):
        #             if pair in group:
        #                 tmp_grp.append(k1)
        #         grouped_key1.append(torch.stack(tmp_grp))
        #     joint_qs_prereduced += grouped_key1


        #     group_jt = []
        #     for k1 in grouped_key1:
        #         # mean encoded q_vals
        #         combined_agents = k1.mean(dim=0) # h
        #         # old
        #         # q_jt = self.linear(combined_agents)
        #         q_jt = self.g(combined_agents)

        #         assert q_jt.shape == (1,), f"Expected {(1,)}, got {q_jt.shape}"
        #         group_jt.append(q_jt)
            
        #     b_qjt.append(torch.stack(group_jt))

        # averaged = []
        # for q in joint_qs_prereduced:
        #     # 'n_pairs' x h
        #     averaged.append(q.mean(dim=0, keepdim=True))
        # averaged = torch.stack(averaged) # 'group' x 1 x h

        # grouped_key2 = []
        # for k2s, close_pairs, groups in zip(key2, batch_close_pairs, batch_groups):
        #     # k2: 'n_pairs' x h
        #     grouped_k2s = []
        #     for group in groups:
        #         tmp_grp = []
        #         for k2, pair in zip(k2s, close_pairs):
        #             if pair in group:
        #                 tmp_grp.append(k2)
        #         grouped_k2s.append(torch.stack(tmp_grp))
        #     grouped_key2 += grouped_k2s

        # b_qjt_alt = []
        # for k2, ave, j in zip(grouped_key2, averaged, joint_qs_prereduced):
        #     tmp = k2 + ave - j/len(j) # len(j) = 'n_pairs'
        #     b_qjt_alt.append(tmp)
        # # b_qjt_alt = key2 + averaged - joint_qs_prereduced/'n_pairs'
        # tmp = []
        # for q in b_qjt_alt:
        #     q = self.phi2(q)
        #     tmp.append(q)
        # b_qjt_alt = tmp

        # return b_qjt, b_qjt_alt

# %% Joint V Network

class VJoint(nn.Module):
    def __init__(self):
        super(VJoint, self).__init__()
        self.hid = LATENT_DIM * 2

        self.v_net = nn.Sequential(
            # nn.Linear(64, 64),
            # nn.LeakyReLU(),
            nn.Linear(self.hid, self.hid),
            nn.LeakyReLU(),
            nn.Linear(self.hid, 1),
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
