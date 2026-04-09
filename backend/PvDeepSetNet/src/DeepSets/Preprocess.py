import pandas as pd
import numpy as np
from itertools import combinations


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


