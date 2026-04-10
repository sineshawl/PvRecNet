import pandas as pd
import numpy as np
import torch
import json
from itertools import combinations
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset



class DataPreprocessor:
    @staticmethod
    def removing_single_episode_patients(data: pd.DataFrame, patient_id='patient_id', episode='episode') -> pd.DataFrame:
        """
        To  run the recurrence analysis, at least 2 episodes are required. So, we have to remove patients with single episode
        data: data
        patient: name of column for the patient_id
        episode: name of column for the episode
        """

        df = data.copy()

        temp = data[[patient_id, episode]].drop_duplicates(ignore_index=True)
        temp = temp.groupby(patient_id)[episode].count().reset_index(name='count')

        removed_id = temp[temp['count'] < 2].patient_id
        return df[~df.patient_id.isin(removed_id)].reset_index(drop=True)
    @staticmethod
    def calculate_population_Allele_frequencies(df, population='country', patient_id='patient_id', episode='episode'):
        """
        Calculate Population Level Allele Frequency
        """
        # We will collect all frequency data here
        pop_aFreq_list = []
        
        # 1. Calculate frequencies per population
        for pop, pop_data in df.groupby(population):
            unique_markers = sorted(pop_data.marker.unique())
            grouped = pop_data.groupby('marker')
            
            for marker, group in grouped:
                if marker not in unique_markers:
                    continue
                    
                # FIX APPLIED: Drop duplicates based on the actual biological event (patient+episode)
                # This prevents "pair-inflation"
                unique_counts = group.drop_duplicates([patient_id, episode, 'allele'])
                
                counts = unique_counts['allele'].value_counts(normalize=True).reset_index()
                counts.columns = ['allele', 'frequency']
                counts['marker'] = marker
                counts[population] = pop # Add population label here
                
                pop_aFreq_list.append(counts)

        # 2. Combine all frequency tables
        all_frequencies = pd.concat(pop_aFreq_list, ignore_index=True)

        # 3. Clean up the original df and merge the calculated frequencies
        output_df = df.drop('frequency', axis=1, errors='ignore').copy()
        output_df = output_df.merge(all_frequencies, on=['marker', 'allele', population], how='inner')

        return output_df
    
    @staticmethod
    def calculate_MOIs(data: pd.DataFrame, patient_id = 'patient_id', episode='episode', marker='marker', allele='allele') -> pd.DataFrame:
        """
        The function to calculcate Multiplicity of Infection (MOI), the maximum number of distinict alleles per sample per marker
        """
        df = data.copy()
        # calculate the number of distinict allele per sample per marker
        temp_df = (
                    data.groupby([patient_id, episode, marker])[allele]
                    .nunique()
                    .reset_index(name="allele_count")
                )
        
        # take the maximum number of alleles across markers per sample

        moi_df = (
                temp_df.groupby([patient_id, episode])["allele_count"]
                .max()
                .reset_index(name='MOI')
                )
        df.drop('MOIs', axis=1, errors='ignore', inplace=True)        

        df = df.merge(moi_df, on=[patient_id, episode], how='inner')
        return df

    @staticmethod    
    def paired_episode_assigner(data: pd.DataFrame, patient_id='patient_id') -> pd.DataFrame:
        """
        This function makes all possible pairs of episodes per patient. 
        So number of episodes = n^2 - n, where n = number of episode
        """
        if 'episode' not in data.columns:
            print('There is no column episode in your dataset')
            return pd.DataFrame()

        all_pairs_list = []

        for indiv, indiv_data in data.groupby(patient_id):
            # Get all unique episodes for this individual
            episodes = sorted(indiv_data['episode'].unique())
            
            # Generate mathematical combinations (n choose 2)
            # combinations(episodes, 2) automatically handles:
            # 1. No self-pairing (i == j is impossible)
            # 2. Uniqueness (if (1,2) is picked, (2,1) is not)
            for pair_order, (ep_i, ep_j) in enumerate(combinations(episodes, 2), 1):
                pair_data = indiv_data[indiv_data['episode'].isin([ep_i, ep_j])].copy()
                
                # Create the unique identifiers
                pair_label = f"{indiv}_P{pair_order}"
                pair_data['pair_order'] = pair_order
                pair_data['sample_id_paired'] = pair_label
                
                all_pairs_list.append(pair_data)

        if not all_pairs_list:
            return pd.DataFrame()
        df = pd.concat(all_pairs_list, ignore_index=True)
        cols = ['sample_id_paired', 'pair_order'] + list(df.columns[:-2])
        df = df[cols]


        # assign episode order: converting every episode into either episode 1 or 2 
        df['episode_order'] = (
            df.groupby('sample_id_paired')['episode']
                .rank(method='dense')
            )

        df.rename(columns={'episode':'true_episode', 'episode_order':'episode'}, inplace=True)

        return df

    @staticmethod
    def build_deepsets_tensors(df, M_max=11, A_max=5, mode='prediction'):
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

            
        # reading allele dictionary
        with open('allele_dict.json', 'r') as file:
            allele_to_id = json.load(file)
        
        # build tensor for target variable only if the mode == evaluation
        y_given = False
        if mode == 'evaluation':
            y_given=True

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
        y      = torch.zeros((N, 3))

        # ----------------------------------------------------
        # 3. Main loop (optimized)
        # ----------------------------------------------------
        for i, (_, df_pair) in enumerate(grouped):
            # ---- pair-level values (constant)
            priors[i] = torch.tensor(df_pair[["prior_C", "prior_L", "prior_I"]].iloc[0].values)
            MOI1 = df_pair.loc[df_pair.episode == 1, 'MOI'].iloc[0] 
            MOI2 = df_pair.loc[df_pair.episode == 2, 'MOI'].iloc[0] 
            
            MOI[i] = torch.tensor([MOI1, MOI2])
            if y_given:
                y[i] = torch.tensor(df_pair[["post_C", "post_L", "post_I"]].iloc[0].values)

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
        final_tensors = {
                "X_alleles": X_alleles, "allele_mask": allele_mask,
                "marker_mask": marker_mask, "priors": priors,
                "MOI": MOI
                }
        if y_given:
            final_tensors['y'] = y

        return final_tensors
    @staticmethod
    def load_tensor(deepset_tensor, mode='prediction', batch_size=32):
        """
        Load tensors with specified batch size
        """

        # TensorDataset
        if mode == 'prediction':
            tensors = TensorDataset(
                deepset_tensor["X_alleles"],
                deepset_tensor["allele_mask"],
                deepset_tensor["marker_mask"],
                deepset_tensor["priors"],
                deepset_tensor["MOI"]
            )
        elif mode == 'evaluation':
            tensors = TensorDataset(
                deepset_tensor["X_alleles"],
                deepset_tensor["allele_mask"],
                deepset_tensor["marker_mask"],
                deepset_tensor["priors"],
                deepset_tensor["MOI"],
                deepset_tensor["y"]
            )


        return DataLoader(tensors, batch_size, shuffle=False)


