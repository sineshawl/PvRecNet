import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Deep Sets model for Pv3Rs surrogate
# =====================================================
class PvDeepSets(nn.Module):
    def __init__(self,
                 n_alleles,        # total number of unique alleles
                 allele_embed_dim=32,
                 marker_embed_dim=64,
                 pair_embed_dim=64,
                 hidden_dim=64,
                 n_classes=3):
        super(PvDeepSets, self).__init__()

        # -------------------------------------------------
        # 1️⃣ Allele embedding (allele ID + frequency)
        # -------------------------------------------------
        self.allele_embedding = nn.Embedding(n_alleles + 1, allele_embed_dim, padding_idx=0)
        self.allele_mlp = nn.Sequential(
            nn.Linear(allele_embed_dim + 1, allele_embed_dim),  # +1 for frequency
            nn.ReLU(),
            nn.Linear(allele_embed_dim, allele_embed_dim),
            nn.ReLU()
        )

        # -------------------------------------------------
        # 2️⃣ Marker-level MLP (after comparing episodes)
        # -------------------------------------------------
        self.marker_mlp = nn.Sequential(
            nn.Linear(allele_embed_dim * 4, marker_embed_dim),
            nn.ReLU(),
            nn.Linear(marker_embed_dim, marker_embed_dim),
            nn.ReLU()
        )

        # -------------------------------------------------
        # 3️⃣ Pair-level MLP (aggregate markers + priors + MOI)
        # -------------------------------------------------
        self.pair_mlp = nn.Sequential(
            nn.Linear(marker_embed_dim + 3 + 2, pair_embed_dim),  # +3 priors +2 MOI
            nn.ReLU(),
            nn.Linear(pair_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, X_alleles, allele_mask, marker_mask, priors, MOI):
        """
        X_alleles : (batch, M_max, 2, A_max, 2)
        allele_mask: same shape as X_alleles[:,:,:,:,0]
        marker_mask: (batch, M_max)
        priors     : (batch, 3)
        MOI        : (batch, 2)
        """
        B, M, E, A, _ = X_alleles.shape

        # ---------------------------------------------
        # 1️⃣ Embed alleles
        # ---------------------------------------------
        allele_ids = X_alleles[..., 0].long()
        freqs = X_alleles[..., 1]

        allele_emb = self.allele_embedding(allele_ids)  # (B,M,E,A,allele_embed_dim)
        allele_input = torch.cat([allele_emb, freqs.unsqueeze(-1)], dim=-1)
        allele_out = self.allele_mlp(allele_input)      # (B,M,E,A,allele_embed_dim)

        # Apply allele mask
        allele_out = allele_out * allele_mask.unsqueeze(-1)

        # ---------------------------------------------
        # 2️⃣ Aggregate alleles → marker embedding per episode
        # ---------------------------------------------
        # Sum pooling over alleles
        marker_ep_emb = allele_out.sum(dim=3)  # (B,M,E,allele_embed_dim)

        # ---------------------------------------------
        # 3️⃣ Compare episodes (ep1 vs ep2)
        # ---------------------------------------------
        ep1 = marker_ep_emb[:, :, 0, :]
        ep2 = marker_ep_emb[:, :, 1, :]

        marker_pair = torch.cat([ep1, ep2, torch.abs(ep1 - ep2), ep1 * ep2], dim=-1)  # (B,M,4*allele_embed_dim)

        # Apply marker-level MLP
        marker_emb = self.marker_mlp(marker_pair)  # (B,M,marker_embed_dim)

        # Masked sum over markers
        marker_emb = marker_emb * marker_mask.unsqueeze(-1)
        pair_emb = marker_emb.sum(dim=1)  # (B, marker_embed_dim)

        # ---------------------------------------------
        # 4️⃣ Concatenate priors + MOI
        # ---------------------------------------------
        pair_emb = torch.cat([pair_emb, priors, MOI], dim=-1)  # (B, marker_dim+4)

        # ---------------------------------------------
        # 5️⃣ Pair-level MLP → logits
        # ---------------------------------------------
        logits = self.pair_mlp(pair_emb)  # (B, n_classes)
        probs = F.softmax(logits, dim=-1)

        return probs

