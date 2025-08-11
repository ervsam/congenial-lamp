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
    def __init__(self, fov, hid_dim: int = LATENT_DIM):
        super().__init__()
        self.layers = 7          # image channels
        self.fov = fov

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
    def __init__(self, fov, USE_NEIGHCOORDS, head_mode: str = "stacked"):
        super(QNetwork, self).__init__()

        self.hid_dim = LATENT_DIM
        self.fov = fov
        self.USE_NEIGH_COORD = USE_NEIGHCOORDS
        self.num_actions = 3
        self.encoder = Encoder(fov, hid_dim=self.hid_dim)

        # Head selection: "stacked" (binary + direction) or "threeway" (single 3-class head)
        allowed_modes = {"stacked", "threeway"}
        if head_mode not in allowed_modes:
            raise ValueError(f"head_mode must be one of {allowed_modes}, got {head_mode}")
        self.head_mode = head_mode

        self.NeighborHeurEncoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.fov * self.fov, self.hid_dim),
            nn.LeakyReLU()
        )

        self.neigh_coord_fc = nn.Sequential(
            nn.Linear(2, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
        )

        self.neigh_out = nn.Sequential(
            nn.Linear(self.hid_dim*2, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
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

        self.bin_fc       = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1),
        )
        self.dir_fc       = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, 2),
        )

        self.neighbor_attn = nn.MultiheadAttention(self.hid_dim, num_heads=4, batch_first=True)

    def forward(self,
                batch_obs,
                batch_neighbor_patches = None,  # Tensor(batch_size, 2, max_neighbor, 1, F, F)
                batch_neigh_coords = None, # Tensor(batch_size, 2, max_neighbor, 2)
                mask = None
            ):

        #### ---------------------------------------------------------------- ##
        #### 0.  Episode-level bookkeeping
        #### ---------------------------------------------------------------- ##
        device   = batch_obs[0].device
        batch_size = len(batch_obs)
        hid_dim = self.hid_dim

        batch_neighbor_patches = batch_neighbor_patches.view(batch_size * 2, -1, 1, self.fov, self.fov)
        batch_neigh_coords = batch_neigh_coords.view(batch_size*2, -1, 2)
        mask = mask.view(batch_size*2, -1)

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
        assert batch_enc.shape == (batch_size*2, hid_dim), f"Expected {(batch_size*2, hid_dim)}, got {batch_enc.shape}"

        # ----- Encode neighbor patches (batched across all episodes and agents) -----
        if batch_neighbor_patches is not None:
            total_agents = batch_neighbor_patches.shape[0]
            all_neighbors_tensor = batch_neighbor_patches
            max_neighbors = all_neighbors_tensor.shape[1]

            # Flatten for CNN: (total_agents * max_neighbors, 1, self.fov, self.fov)
            flat_neighbors = all_neighbors_tensor.view(-1, 1, self.fov, self.fov)
            assert flat_neighbors.shape == (total_agents * max_neighbors, 1, self.fov, self.fov), f"expected {(total_agents * max_neighbors, 1, self.fov, self.fov)}, but got {flat_neighbors.shape}"

            neighbor_embeds = self.NeighborHeurEncoder(flat_neighbors)  # (total_agents * max_neighbors, hid_dim)

            if self.USE_NEIGH_COORD:
                pad_neigh_coord = batch_neigh_coords

                flat_coords = pad_neigh_coord.view(-1, 2)
                coords_embeds = self.neigh_coord_fc(flat_coords)
                # coords_embeds: (total_agents * max_neighbors, hid_dim)
                neighcoords_embeds = torch.cat([neighbor_embeds, coords_embeds], dim=1)
                neighbor_embeds = self.neigh_out(neighcoords_embeds)

            neighbor_embeds = neighbor_embeds.view(total_agents, max_neighbors, hid_dim)

            # ----------- Batched attention for all agents -------------
            agent_embed_q = batch_enc.unsqueeze(1)          # (total_agents, 1, H)
            assert agent_embed_q.shape == (batch_size*2, 1, hid_dim), f"Expected {(batch_size*2, 1, hid_dim)}, got {agent_embed_q.shape}"
            neighbor_embeds_kv = neighbor_embeds                          # (total_agents, max_neighbors, H)
            assert neighbor_embeds_kv.shape == (batch_size*2, max_neighbors, hid_dim), f"Expected {(batch_size*2, max_neighbors, hid_dim)}, got {neighbor_embeds_kv.shape}"

            attn_out, _ = self.neighbor_attn(
                agent_embed_q,
                neighbor_embeds_kv,
                neighbor_embeds_kv,
                key_padding_mask=mask
            )  # (total_agents, 1, H)
            fused_embeds = agent_embed_q.squeeze(1) + attn_out.squeeze(1)  # (total_agents, H)
            assert fused_embeds.shape == (batch_size*2, hid_dim), f"Expected {(batch_size*2, hid_dim)}, got {fused_embeds.shape}"

            batch_enc = fused_embeds

        assert batch_enc.shape == (batch_size*2, hid_dim), f"Expected {(batch_size*2, hid_dim)}, got {batch_enc.shape}"
        
        #### ---------------------------------------------------------------- ##
        #### 2.  Build pair encodings + utility head for each episode
        #### ---------------------------------------------------------------- ##

        # pair_enc = torch.cat([h_a, h_b], dim=-1)     # (n_pairs, 2H)
        pair_enc_per_ep = batch_enc.view(batch_size, 2 * hid_dim)

        # Choose head based on configuration
        if self.head_mode == "stacked":
            # Binary + direction heads
            bin_logits = self.bin_fc(pair_enc_per_ep).squeeze(-1)   # (B,)
            dir_logits = self.dir_fc(pair_enc_per_ep)                # (B,2)
            return pair_enc_per_ep, bin_logits, dir_logits
        else:  # "threeway"
            # Single 3-class head
            class_logits = self.qnet(pair_enc_per_ep)                # (B,3)
            return pair_enc_per_ep, class_logits
