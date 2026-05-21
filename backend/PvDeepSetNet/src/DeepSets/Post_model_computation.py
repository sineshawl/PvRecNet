import pandas as pd
import numpy as np

class Post_Model_Computation:
    @staticmethod
    def compute_pv3rs_marginal_probabilities(results_df):
        """
        Complete Python implementation of the Pv3Rs approx-joint marginalization logic.
        
        1. Assigns pair_source based on the group cumcount index (+1).
        2. Sequentially evaluates each true recurrence episode (j).
        3. Normalizes and rounds the probabilities to 4 decimal places.
        """
        # ====================================================================================
        # STEP 1: Assign Source Position Index Using Your Cumcount Logic
        # ====================================================================================
        df = results_df.sort_values(['patient_id', 'true_episode']).copy()
        
        # This identifies if the row is comparing against the 1st, 2nd, or 3rd previous episode
        df['pair_source'] = df.groupby(['patient_id', 'true_episode']).cumcount() + 1    
        df = df.sort_values('pair_id')
        
        marginal_results = []
        
        # ====================================================================================
        # STEP 2: Sequential Marginal Aggregation Engine
        # ====================================================================================
        for pid, group in df.groupby('patient_id'):
            
            # Get unique episodes in chronological order
            episodes = sorted(group['true_episode'].unique())
            n_epi = len(episodes)
            
            # We start calculating from the first recurrence (index 1)
            for idx in range(1, n_epi):
                curr_epi = episodes[idx]       # Current recurrence episode 'j'
                prev_epi = episodes[idx - 1]   # Immediate predecessor 'j-1'
                
                # Filter all pairs where 'true_episode' is our current episode
                curr_recurrence_pairs = group[group['true_episode'] == curr_epi]
                
                if curr_recurrence_pairs.empty:
                    continue
                    
                # --- 1. Compute C_unnorm (Consecutive pair constraint) ---
                # Rule: Only look at the immediate predecessor index
                c_row = curr_recurrence_pairs[curr_recurrence_pairs['pair_source'] == prev_epi]
                C_unnorm = c_row['prob_C'].iloc[0] if not c_row.empty else 0.0
                
                # --- 2. Compute L_unnorm & Track Source (Max operator across all i < j) ---
                max_l_idx = curr_recurrence_pairs['prob_L'].idxmax()
                L_unnorm = curr_recurrence_pairs.loc[max_l_idx, 'prob_L']
                L_source = curr_recurrence_pairs.loc[max_l_idx, 'pair_source']
                
                # --- 3. Compute I_unnorm (Min operator across all i < j) ---
                I_unnorm = curr_recurrence_pairs['prob_I'].min()
                
                # --- 4. Apply the Pv3Rs Conditional Correction Rule ---
                if C_unnorm > 0.5 and I_unnorm < 0.001:
                    L_unnorm = c_row['prob_L'].iloc[0] if not c_row.empty else L_unnorm
                    L_source = prev_epi
                    
                # --- 5. Total Normalization & Formatted Rounding ---
                total_mass = C_unnorm + L_unnorm + I_unnorm
                
                if total_mass > 0:
                    C_marg = round(C_unnorm / total_mass, 4)
                    L_marg = round(L_unnorm / total_mass, 4)
                    I_marg = round(I_unnorm / total_mass, 4)
                else:
                    C_marg, L_marg, I_marg = 0.0, 0.0, 0.0
                
                # Append structured row for this specific recurrence outcome
                marginal_results.append({
                    'sample_id': pid + '_' + str(int(curr_epi)),
                    'patient_id': pid,
                    'episode': int(curr_epi),
                    'relapse_source_epi': int(L_source),
                    'prob_C': C_marg,
                    'prob_L': L_marg,
                    'prob_I': I_marg
                })
                
        return pd.DataFrame(marginal_results)


    @staticmethod
    def result_post_processing(results_df, calc_NU_post = True, approx_joint = True):
        """
        result_df: PvDeepSet predicted results
        calc_NU_post: calculate non uniform posterior (refine the posterior if the priors were not default)
        approx_joint: approximate marginal probabilities from perwise results
        """
        if calc_NU_post:
            # unique_result = results_df[['pair_id', 'prior_C', 'prior_L', 'prior_I', 'prob_C', 'prob_L', 'prob_I']].drop_duplicates(ignore_index=True)
            # 1. Align column names for math operations
            # We create a temporary copy to ensure we multiply the right classes
            df_priors = results_df[['prior_C', 'prior_L', 'prior_I']].rename(columns={'prior_C': 'C', 'prior_L': 'L', 'prior_I': 'I'})
   
            df_post_uni = results_df[['prob_C', 'prob_L', 'prob_I']].rename(columns={'prob_C': 'C', 'prob_L': 'L', 'prob_I': 'I'})

            # 2. Numerator: Multiply P_uniform(s|y) * P(s)
            # Pandas handles row-matching automatically by index
            numerator = df_post_uni * df_priors

            # 3. Denominator: Sum across the columns for each row (Normalization)
            denominator = numerator.sum(axis=1)

            # 4. Final Recovery: Divide numerator by denominator
            # axis=0 ensures the division happens row by row
            df_post_non_uniform = numerator.divide(denominator, axis=0)

            df_post_non_uniform['C'] = df_post_non_uniform['C'].round(4)
            df_post_non_uniform['L'] = df_post_non_uniform['L'].round(4)
            df_post_non_uniform['I'] = df_post_non_uniform['I'].round(4)
    
            results_df['prob_C'] = df_post_non_uniform['C'].values
            results_df['prob_L'] = df_post_non_uniform['L'].values
            results_df['prob_I'] = df_post_non_uniform['I'].values

        if approx_joint:
            results_df =  Post_Model_Computation.compute_pv3rs_marginal_probabilities(results_df=results_df)
        return results_df