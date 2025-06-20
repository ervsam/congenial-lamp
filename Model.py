import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum

LATENT_DIM = 256

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
        self.fov = 21

        self.conv = nn.Sequential(
            nn.Conv2d(self.layers, 32, 3, padding=1),
            nn.LeakyReLU(),
            ResBlock(32),                             # <- keep two res-blocks
            ResBlock(32),
            ResBlock(32),
            # nn.Dropout2d(p=0.10),                     # not 0.20
            nn.Conv2d(32, 16, 1), 
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16*self.fov*self.fov, hid_dim),
            nn.LeakyReLU()
        )
        self.coord_fc = nn.Linear(2, 64)
        self.out      = nn.Linear(hid_dim + 64, hid_dim)

    def forward(self, x, coords):
        h_img   = self.conv(x)
        h_coord = self.coord_fc(coords)
        h       = torch.cat([h_img, h_coord], dim=-1)
        return self.out(h)

# %% Q-Network (agent utility network)
class QNetwork(nn.Module):
    def __init__(self, fov):
        super(QNetwork, self).__init__()
        self.CONCAT = True

        self.hid_dim = LATENT_DIM
        self.fov = fov
        self.num_actions = 3
        self.encoder = Encoder(hid_dim=self.hid_dim)

        self.NeighborHeurEncoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.fov * self.fov, self.hid_dim),
            nn.LeakyReLU()
        )

        # utility net (concatenation path)
        self.qnet = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.hid_dim, self.num_actions),
        )

        # New: Neighbor attention module (can use MultiheadAttention)
        self.neighbor_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True)

    def forward(self,
                batch_obs: list[torch.Tensor],
                batch_close_pairs: list[list[tuple[int,int]]],
                batch_neighbor_patches: list[list[torch.Tensor]] = None,  # list of (num_agents, num_neighbors, C, F, F)
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        #### ---------------------------------------------------------------- ##
        #### 0.  Episode-level bookkeeping
        #### ---------------------------------------------------------------- ##
        device   = batch_obs[0].device
        batch_size = len(batch_obs)
        hid_dim = self.hid_dim

        #### ---------------------------------------------------------------- ##
        #### 1.  Flatten → encode every agent once
        #### ---------------------------------------------------------------- ##
        # batch_obs: batch_size, 2, C, F, F
        batch_obs = batch_obs.view(batch_size*2, 8, self.fov, self.fov)

        batch_coordinates = batch_obs[:, -1, 0, 0:2]    # (ΣN_i, 2)
        assert batch_coordinates.shape == (batch_size*2, 2), f"Expected {(batch_size*2, 2)}, got {batch_coordinates.shape}"

        batch_obs = batch_obs[:, :-1]    # (ΣN_i, C-1, F, F)
        assert batch_obs.shape == (batch_size*2, 7, self.fov, self.fov), f"Expected {(batch_size*2, 7, self.fov, self.fov)}, got {batch_obs.shape}"

        batch_enc   = self.encoder(batch_obs, batch_coordinates)     # (batch_size*2, h)
        assert batch_enc.shape == (batch_size*2, hid_dim)

        # ----- Encode neighbor patches (batched across all episodes and agents) -----
        if batch_neighbor_patches is not None:
            # Flatten all agent embeddings, all neighbors, and store mapping
            total_agents = len(batch_enc)

            all_neighbor_patches = []
            neighbor_lens = []    # number of neighbors per agent
            for neighbor_patches in batch_neighbor_patches:
                patches_1 = neighbor_patches[0].to(device)
                neighbor_lens.append(len(patches_1))
                all_neighbor_patches.append(patches_1)

                patches_2 = neighbor_patches[1].to(device)
                neighbor_lens.append(len(patches_2))
                all_neighbor_patches.append(patches_2)

            # Pad neighbor lists to max_neighbors and stack
            max_neighbors = max(neighbor_lens)
            padded_neighbors = []
            for patches in all_neighbor_patches:
                n = len(patches)
                if n < max_neighbors:
                    pad = torch.zeros((max_neighbors - n, 1, self.fov, self.fov), device=device)
                    padded_neighbors.append(torch.cat([patches, pad], dim=0))
                else:
                    padded_neighbors.append(patches)
            # shape: (total_agents, max_neighbors, 1, self.fov, self.fov)
            all_neighbors_tensor = torch.stack(padded_neighbors, dim=0)
            assert all_neighbors_tensor.shape == (batch_size*2, max_neighbors, 1, self.fov, self.fov)

            # Flatten for CNN: (total_agents * max_neighbors, 1, self.fov, self.fov)
            flat_neighbors = all_neighbors_tensor.view(-1, 1, self.fov, self.fov)
            neighbor_embeds = self.NeighborHeurEncoder(flat_neighbors)  # (total_agents * max_neighbors, hid_dim)
            neighbor_embeds = neighbor_embeds.view(total_agents, max_neighbors, hid_dim)

            all_agent_embeds_tensor = batch_enc  # (total_agents, hid_dim)

            # ----------- Build mask for attention -------------
            mask = torch.zeros((total_agents, max_neighbors), dtype=torch.bool, device=device)
            for i, n in enumerate(neighbor_lens):
                if n < max_neighbors:
                    mask[i, n:] = True

            # ----------- Batched attention for all agents -------------
            agent_embed_q = all_agent_embeds_tensor.unsqueeze(1)          # (total_agents, 1, H)
            assert agent_embed_q.shape == (batch_size*2, 1, hid_dim)
            neighbor_embeds_kv = neighbor_embeds                          # (total_agents, max_neighbors, H)
            assert neighbor_embeds_kv.shape == (batch_size*2, max_neighbors, hid_dim)

            attn_out, _ = self.neighbor_attn(
                agent_embed_q,
                neighbor_embeds_kv,
                neighbor_embeds_kv,
                key_padding_mask=mask
            )  # (total_agents, 1, H)
            fused_embeds = attn_out.squeeze(1)  # (total_agents, H)
            assert fused_embeds.shape == (batch_size*2, hid_dim)

            # Optionally set n=0 agents to their own embedding
            for i, n in enumerate(neighbor_lens):
                if n == 0:
                    fused_embeds[i] = all_agent_embeds_tensor[i]

            batch_enc = fused_embeds

        assert batch_enc.shape == (batch_size*2, hid_dim)
        
        #### ---------------------------------------------------------------- ##
        #### 2.  Build pair encodings + utility head for each episode
        #### ---------------------------------------------------------------- ##

        # pair_enc = torch.cat([h_a, h_b], dim=-1)     # (n_pairs, 2H)
        pair_enc_per_ep = batch_enc.view(batch_size, 2 * hid_dim)

        # pair_q   = self.qnet(pair_enc)               # (n_pairs, 2)
        qvals_per_ep = self.qnet(pair_enc_per_ep)

        # assert len(qvals_per_ep) == batch_size, f"Expected {batch_size}, got {len(qvals_per_ep)}"
        # assert qvals_per_ep[0].shape[1] == self.num_actions, f"Expected {self.num_actions}, got {qvals_per_ep[0].shape[1]}"
        return pair_enc_per_ep, qvals_per_ep


class GroupAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    def forward(self, x):
        # x: (seq_len, batch, embed_dim) -- for each group, batch=1
        x = x.unsqueeze(1)  # (seq_len, 1, embed_dim)
        out, _ = self.attn(x, x, x)
        return out.mean(dim=0)  # (1, embed_dim) mean over seq


# %% Joint Q-Network
class QJoint(nn.Module):
    def __init__(self):
        super(QJoint, self).__init__()

        self.hid = LATENT_DIM
        self.in_dim = self.hid * 2 + 2        # encA ⊕ encB ⊕ one-hot

        self.group_attn = GroupAttention(self.hid * 2, num_heads=2)

        # φ₁  (key1) – encodes each pair
        self.phi1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.hid, self.hid * 2),
        )

        # g – combines the mean of φ₁ into a scalar Qjt
        self.g = nn.Sequential(
            nn.Linear(self.hid * 2, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.hid, 1),
        )

        # φ₂  (key2) – value for alternative head
        self.phi2 = nn.Sequential(
            nn.Linear(self.hid * 2, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, self.hid),
            nn.ELU(),
            nn.Linear(self.hid, 2),          # outputs per-pair Q-vector
        )

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
            # key1_mean = scatter_mean(key1, group_index, dim=0)              # (G_i, H)
            key1_mean = []
            for g in groups:
                idxs = [pair2idx[p] for p in g]
                key1_g = key1[idxs]      # (n_pairs_in_group, H)

                # DELETE
                # print("shape of key1_g:", key1_g.shape)
                # if key1_g.shape[0] == 1:
                #     print("shape of key1_g:", key1_g.shape)
                #     print("key1_g:", key1_g)
                    
                group_repr = self.group_attn(key1_g)  # (1, H)

                # DELETE
                # if key1_g.shape[0] == 1:
                #     print("group_repr:", group_repr)

                key1_mean.append(group_repr.squeeze(0))
            key1_mean = torch.stack(key1_mean, dim=0)  # (G_i, H)

            # DELETE
            # if len(groups) > 1:
            #     print("groups from Qjoint:", groups)
            #     print("group_index:", group_index)
            #     print("key1_mean:", key1_mean)
            #     print("key1_mean", key1_mean[1])
            #     print("key1:", key1[pair2idx[groups[1][0]]])
            #     print("key1_mean == key1:", key1_mean[1] == key1[pair2idx[groups[1][0]]])
            #     quit()

            # scalar Qjt per group   (Eq. 9)
            q_jt = self.g(key1_mean).squeeze(-1)  

            # ------------------------------------------------------------------ #
            # 1-c  Alternative head  (Eq. 10   k₂ + k̄₁ − k₁/|g|)
            # ------------------------------------------------------------------ #
            # Need broadcasted tensors:
            #   k₂[p],   mean_key1_of_group[p],   key1[p]/|g|
            #
            k2            = key2                                             # (P_i, 2H)

            counts = scatter_sum(torch.ones(len(pairs), device=key1.device),
                                group_index, dim=0)          # → (G_i,)
            # expand counts back to per-pair
            count_per_pair = counts[group_index].unsqueeze(-1) # → (P_i,1)
            k1_div_len_g = key1 / count_per_pair               # → (P_i, D)

            # print("counts", counts)
            # print("count_per_pair", count_per_pair[0])            
            # print("k1_div_len_g", k1_div_len_g[0])
            # print("key1", key1[0])
            # quit()

            kbar1         = key1_mean[group_index]                           # (P_i, H)
            alt_val       = k2 + kbar1 - k1_div_len_g                        # (P_i,2H)  (2H if φ₂ expects that)
            alt_q         = self.phi2(alt_val)                               # (P_i, 2)
            
            # reorganise alt_q into (G_i, n_pairs_in_g, 2) by explicit group membership
            # map each pair tuple to its row index
            alt_per_g = []
            for group in groups:
                idxs = [pair2idx[p] for p in group]
                alt_per_g.append(alt_q[idxs])

            out_qjt.append(q_jt.unsqueeze(-1))        # keep shape (G_i,1) to match old code
            out_qjt_alt += alt_per_g
            # out_qjt_alt.append(alt_per_g)

        return out_qjt, out_qjt_alt 



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
