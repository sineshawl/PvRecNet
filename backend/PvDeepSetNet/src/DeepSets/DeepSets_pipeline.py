import json
import torch
from sklearn.metrics import r2_score
from .Preprocess import DataPreprocessor
from .Model_result import PvDeepSet_Model



def run_prediction(data, mode='prediction'):

    #-------------------------------------#
    # 1. perform basic preprocess
    #-------------------------------------#
    if ('loci' in data.columns) and (not 'marker' in data.columns):
        data.rename(columns={'loci':'marker'}, inplace=True)
    if ('haplotype' in data.columns) and (not 'allele' in data.columns):
        data.rename(columns={'haplotype':'allele'}, inplace=True)
    if ('region' in data.columns) and (not 'country' in data.columns):
        data.rename(columns={'region':'country'}, inplace=True)


    if mode == 'prediction':
        req_column = ['patient_id', 'episode', 'marker', 'allele', 'country']
        data, meta_df = (
            data
            .pipe(DataPreprocessor.removing_single_episode_patients)
            .pipe(DataPreprocessor.calculate_population_Allele_frequencies, population='country')
            .pipe(DataPreprocessor.calculate_MOIs)
            .pipe(DataPreprocessor.paired_episode_assigner)
        )
        # meta_df = data[['sample_id_paired', 'patient_id', 'true_episode', 'prior_C', 'prior_L', 'prior_I']].drop_duplicates().reset_index(drop=True)
        tensors = DataPreprocessor.build_deepsets_tensors(df=data, mode=mode)
        loaded_data = DataPreprocessor.load_tensor(tensors, mode)

        return PvDeepSet_Model.run_prediction(meta_df=meta_df, 
                                        pair_id=tensors['pair_id'],
                                        data_loader = loaded_data, 
                                        basic_summary=True)





