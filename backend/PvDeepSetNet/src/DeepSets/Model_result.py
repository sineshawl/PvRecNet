import os
import json
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score
from DeepSets import PvDeepSets, Post_Model_Computation, PvRecNet_Plots
from sklearn.metrics import classification_report, confusion_matrix
import re




class PvDeepSet_Model:
    @staticmethod 
    def load_model_state():
        #-------------------------------------#
        # 2. Loading model state dict
        #-------------------------------------#

        # reading allele dictionary
        # Get the directory where Model_result.py is located
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'allele_dict.json')

        with open(file_path, 'r') as file:
            allele_to_id = json.load(file)

        # Define the device
        device = torch.device("cpu")

        # Initialize the model with the same parameters used during training
        # Make sure len(allele_to_id) matches the training setup!
        n_alleles = len(allele_to_id) 
        model = PvDeepSets(n_alleles=n_alleles)
        print(n_alleles)
        # Load the weights (state_dict)
        # Get the directory where Preprocess.py is located
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'pv_deepsets_weights20260425_154256.pth')
        model.load_state_dict(torch.load(file_path, weights_only=True, map_location=device))

        return device, model
   

    @staticmethod
    def get_model_results(model, dataloader, device, threshold=0.5, mode='prediction'):
        model.eval()
        all_predicted = []
        all_actual = []

        with torch.no_grad():
            for batch in dataloader:
                # Flexible unpacking: if evaluating, 'batch' has one extra item (y)
                if mode  == 'evaluation':
                    X_alleles, allele_mask, marker_mask, MOI, y = batch
                    true_labels = torch.argmax(y, dim=1)
                    all_actual.append(y.cpu().numpy())
                elif mode == 'prediction':
                    X_alleles, allele_mask, marker_mask, MOI = batch

                # Model Forward Pass
                y_pred = model(X_alleles.to(device), allele_mask.to(device), 
                            marker_mask.to(device), MOI.to(device))
                
                probs = y_pred # Assuming Softmax is handled inside or by thresholding

                all_predicted.append(y_pred.cpu().numpy())
        
        # Consolidate results
        pred_stack = np.vstack(all_predicted)
        
        if mode == 'evaluation':
            actual_stack = np.vstack(all_actual)
            return actual_stack, pred_stack
        
        return pred_stack

    @staticmethod
    def run_prediction(meta_df,
                       pair_id,
                       data_loader,
                       basic_summary=True):
        
        # 1. call a function that load saved model state
        device, model = PvDeepSet_Model.load_model_state()

        # 2. get model prediction result 
        # Shape: [N, 3] -> Column 0: C, Column 1: L, Column 2: I
        predicted_probs = PvDeepSet_Model.get_model_results(model=model, dataloader=data_loader, device=device, mode='prediction')

        # 3. Merge IDs with Probabilities
        # Create a DataFrame from the results
        results_df = pd.DataFrame(
            predicted_probs, 
            columns=['prob_C', 'prob_L', 'prob_I']
        )

        # Insert the pair_id as the first column
        results_df.insert(0, 'pair_id', pair_id)

        results_df = results_df.merge(meta_df, left_on='pair_id', right_on='sample_id_paired', how='inner')
        # model result post processing (prior refinement & marginal probabilities calculations)
        results_df = Post_Model_Computation.result_post_processing(results_df, calc_NU_post=True, approx_joint=True)


        # 4. Hardclass Classification with Threshold Logic
        def classify_row(row, cols=['prob_C', 'prob_L', 'prob_I']):
            probs = [row[cols[0]], row[cols[1]], row[cols[2]]]
            max_p = max(probs)
            
            # Check if the highest probability meets the 0.5 threshold
            if max_p < 0.5:
                return "undefined"
            
            # Otherwise, find the name of the max class
            class_names = ['C', 'L', 'I']
            return class_names[np.argmax(probs)]

        # Apply the logic to create the new column
        results_df['predicted_class'] = results_df.apply(classify_row, axis=1)

        # ploting the recurrence classification results 
        donut_plot = PvRecNet_Plots.plot_donut_count(results_df)
        probability_dist_plot = PvRecNet_Plots.plot_probability_dist(results_df)       
        

        columns = ['pair_id', 'patient_id', 'true_episode', 'prob_C', 'prob_L', 'prob_I', 'predicted_class']
        results = {
                'results_table':results_df,
                'donut_plot': donut_plot,
                'distribution_plot': probability_dist_plot
                }

        return results
   