import torch
import pandas as pd
import json
from collections import defaultdict

def build_deepsets_tensors(df, M_max = 11, A_max = 5, purpose='prediction', prior='not_given'):
    """
    df =  dataframe
    M_max = the maximum number of unique marker within the dataset (default 11 <- used during training)
    A_max = the maximum number of unique allele per marker (default 5 <- used during training)
    allele_to_id = alleles with their corresponding ids
    Optimized Deep Sets preprocessing for Pv3Rs surrogate
    """
    # checking the existance of prior columns, otherwise the default values will be assigned
    required_cols = {'prior_C', 'prior_L', 'prior_I'}
    if not required_cols.issubset(df.columns):
        df['prior_C']=df['prior_L']=df['prior_I']=1/3
        

        

    with open('allele_dict.json', 'r') as file:
        allele_to_id = json.load(file)
    

    # ----------------------------------------------------
    # 1. Encode categorical variables once
    # ----------------------------------------------------
    marker_to_id = {m: i + 1 for i, m in enumerate(df.marker.unique())}
    allele_to_id = allele_to_id

    grouped = list(df.groupby("sample_id_paired")) #computationally intensive
    N = len(grouped)

    # ----------------------------------------------------
    # 2. Allocate tensors
    # ----------------------------------------------------
    X_alleles   = torch.zeros((N, M_max, 2, A_max, 2))
    allele_mask = torch.zeros((N, M_max, 2, A_max), dtype=torch.bool)
    marker_mask = torch.zeros((N, M_max), dtype=torch.bool)

    priors = torch.zeros((N, 3))
    MOI    = torch.zeros((N, 2))

    # ----------------------------------------------------
    # 3. Main loop (optimized)
    # ----------------------------------------------------
    for i, (_, df_pair) in enumerate(grouped):
        # ---- pair-level values (constant)
        priors[i] = torch.tensor(df_pair[["prior_C", "prior_L", "prior_I"]].iloc[0].values)
        MOI1 = df_pair.loc[df_pair.episode == 1, 'MOI'].iloc[0] 
        MOI2 = df_pair.loc[df_pair.episode == 2, 'MOI'].iloc[0] 
        
        MOI[i] = torch.tensor([MOI1, MOI2])

        # ---- pre-group alleles by (marker, episode)
        allele_dict = defaultdict(list)
        for row in df_pair.itertuples():
            allele_dict[(row.marker, row.episode)].append((row.allele, row.frequency))

        # ---- marker union
        markers = sorted({m for (m, _) in allele_dict.keys()})[:M_max]

        for m_idx, marker in enumerate(markers):
            marker_mask[i, m_idx] = True

            for e_idx, episode in enumerate([1, 2]):
                alleles = allele_dict.get((marker, episode), [])

                for a_idx, (allele, freq) in enumerate(alleles[:A_max]):
                    X_alleles[i, m_idx, e_idx, a_idx, 0] = allele_to_id[allele]
                    X_alleles[i, m_idx, e_idx, a_idx, 1] = freq
                    allele_mask[i, m_idx, e_idx, a_idx] = True

    return {
        "X_alleles": X_alleles,
        "allele_mask": allele_mask,
        "marker_mask": marker_mask,
        "priors": priors,
        "MOI": MOI,
    }
