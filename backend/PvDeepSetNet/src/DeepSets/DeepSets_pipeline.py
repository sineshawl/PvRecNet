import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from DeepSets import DataPreprocessor, build_deepsets_tensors




def run_Prediction(data, mode='prediction'):

    # perform basic preprocess
    preprocessed_tensor = (
        data
        .pipe(DataPreprocessor.removing_single_episode_patients)
        .pipe(DataPreprocessor.calculate_population_Allele_frequencies, population='country')
        .pipe(DataPreprocessor.calculate_MOIs)
        .pipe(DataPreprocessor.paired_episode_assigner)
        .pipe(DataPreprocessor.build_deepsets_tensors, mode=mode)
    )
    loaded_data = DataPreprocessor.load_tensor(preprocessed_tensor, mode)
    return loaded_data


