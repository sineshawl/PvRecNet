import json
import torch
from sklearn.metrics import r2_score
from DeepSets import DataPreprocessor, PvDeepSet_model



def run_prediction(data, mode='prediction'):

    #-------------------------------------#
    # 1. perform basic preprocess
    #-------------------------------------#
   
    preprocessed_data, meta_df = (
        data
        .pipe(DataPreprocessor.removing_single_episode_patients)
        .pipe(DataPreprocessor.calculate_population_Allele_frequencies, population='country')
        .pipe(DataPreprocessor.calculate_MOIs)
        .pipe(DataPreprocessor.paired_episode_assigner)
    )
    tensors = DataPreprocessor.build_deepsets_tensors(df=preprocessed_data, mode=mode)
    loaded_data = DataPreprocessor.load_tensor(tensors, mode)

    #-------------------------------------#
    # 2. model prediction and evaluation
    #-------------------------------------#

    if mode == 'prediction':
        return PvDeepSet_model.run_prediction(meta_df=meta_df, 
                                              pair_id=tensors['pair_id'],
                                              data_loader = loaded_data, 
                                              basic_summary=True)
    elif mode == 'evaluation':
        return PvDeepSet_model.run_evaluation(meta_df=meta_df,
                                              pair_id=tensors['pair_id'],
                                              data_loader=loaded_data, 
                                              probability_plot = True, 
                                              metrics=True, 
                                              confusion_matrix=True, 
                                              basic_summary=True)
    else:
        return 'incorrect mode'
    

