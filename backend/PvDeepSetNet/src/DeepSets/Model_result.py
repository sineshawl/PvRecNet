import os
import json
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score
from DeepSets import PvDeepSets
from sklearn.metrics import classification_report, confusion_matrix




class PvDeepSet_model:
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
        file_path = os.path.join(base_path, 'pv_deepsets_weights20260402.pth')
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
                    X_alleles, allele_mask, marker_mask, priors, MOI, y = batch
                    true_labels = torch.argmax(y, dim=1)
                    all_actual.append(y.cpu().numpy())
                elif mode == 'prediction':
                    X_alleles, allele_mask, marker_mask, priors, MOI = batch

                # Model Forward Pass
                y_pred = model(X_alleles.to(device), allele_mask.to(device), 
                            marker_mask.to(device), priors.to(device), MOI.to(device))
                
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
        device, model = PvDeepSet_model.load_model_state()

        # 2. get model prediction result 
        # Shape: [N, 3] -> Column 0: C, Column 1: L, Column 2: I
        predicted_probs = PvDeepSet_model.get_model_results(model=model, dataloader=data_loader, device=device, mode='prediction')

        # 3. Merge IDs with Probabilities
        # Create a DataFrame from the results
        results_df = pd.DataFrame(
            predicted_probs, 
            columns=['prob_C', 'prob_L', 'prob_I']
        )

        # Insert the pair_id as the first column
        results_df.insert(0, 'pair_id', pair_id)

        # 4. Hardclass Classification with Threshold Logic
        def classify_row(row):
            probs = [row['prob_C'], row['prob_L'], row['prob_I']]
            max_p = max(probs)
            
            # Check if the highest probability meets the 0.5 threshold
            if max_p < 0.5:
                return "undefined"
            
            # Otherwise, find the name of the max class
            class_names = ['C', 'L', 'I']
            return class_names[np.argmax(probs)]

        # Apply the logic to create the new column
        results_df['predicted_class'] = results_df.apply(classify_row, axis=1)

        dist_df = results_df['predicted_class'].value_counts().reset_index()
        dist_df.columns = ['class', 'count']
    

        # fig = px.bar(
        #     dist_df, x='class', y='count', 
        #     color='class', title="Classification Distribution",
        #     template="plotly_white"
        # )
        def plot_donut_count(df):
            # 1. Prepare the data
            # Assuming 'df' is your processed dataframe from the previous step
            counts_df = df['predicted_class'].value_counts().reset_index()
            counts_df.columns = ['Class', 'Count']
            map_rec = {'C':'C (Recrudescence)', 'L': 'L (Relapse)', 'I':'I (Reinfection)'}

            counts_df['Class'] = counts_df['Class'].map(map_rec)

            # 2. Create the Donut Chart
            fig = px.pie(
                counts_df, 
                values='Count', 
                names='Class', 
                hole=0.5, # This creates the "donut" effect
                color='Class',
                # Keeping colors consistent: Orange (C), Yellow (L), Teal (I)
                color_discrete_map={
                    'C (Recrudescence)': '#ff4d00', 
                    'L (Relapse)': '#e6b400', 
                    'I (Reinfection)': '#1fb3c4'
                },
                template="plotly_white"
            )

            # 3. Refine the layout to match your previous style
            fig.update_layout(
                margin=dict(l=20, r=20, t=60, b=20),
                height=500,
                paper_bgcolor='white',
                # Horizontal legend at the top
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    title=None
                )
            )

            # 4. Enhance the traces (labels and percentages)
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label', # Shows the Class name and %
                marker=dict(line=dict(color='#FFFFFF', width=2)) # Adds a small white gap between segments
            )

            return fig


        def plot_probability_dist(df):

            df = df.copy()
            df = df[['pair_id', 'prob_C', 'prob_L','prob_I','predicted_class']].drop_duplicates().reset_index(drop=True)

            # Grouped and sorted blocks
            df_C = df[df.predicted_class == 'C'].sort_values('prob_C', ascending=False).reset_index(drop=True)
            df_L = df[df.predicted_class == 'L'].sort_values('prob_L', ascending=False).reset_index(drop=True)
            df_I = df[df.predicted_class == 'I'].sort_values('prob_I', ascending=False).reset_index(drop=True)

            df_sorted = pd.concat([df_C, df_L, df_I], ignore_index=True)

            df_long = df_sorted.melt(
                id_vars=['pair_id'], 
                value_vars=['prob_C', 'prob_L', 'prob_I'],
                var_name='Probability_Type', 
                value_name='Probability'
            )

            fig = px.bar(
                df_long, 
                x="pair_id", 
                y="Probability", 
                color="Probability_Type",
                color_discrete_sequence=['#ff4d00', '#e6b400', '#1fb3c4'],
                category_orders={"pair_id": df_sorted['pair_id'].tolist()},
                template="plotly_white" # Sets white background and clean gridlines
            )

            fig.update_layout(
                title=None,
                margin=dict(l=50, r=20, t=60, b=50), # Adjusted for axis labels
                height=500,
                paper_bgcolor='white',
                plot_bgcolor='white',
                
                # Horizontal legend at the top
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    title=None
                ),
                
                xaxis=dict(
                    title="Samples",      # Restored the label
                    showticklabels=False, # Keeps specific IDs hidden for cleanliness
                    ticks="",
                    linecolor='black',    # Adds a crisp base line
                    showgrid=False
                ),
                
                yaxis=dict(
                    title="Probability",
                    range=[0, 1],
                    tickformat=".1f",
                    linecolor='black',
                    gridcolor='lightgrey' # Subtle grid for probability reference
                ),
                
                bargap=0
            )

            return fig
        
        donut_plot = plot_donut_count(results_df)
        probability_dist_plot = plot_probability_dist(results_df)       
        
        results_df = results_df.merge(meta_df, left_on='pair_id', right_on='sample_id_paired', how='inner')

        columns = ['pair_id', 'patient_id', 'true_episode', 'prob_C', 'prob_L', 'prob_I', 'predicted_class']
        results = {
                'results_table':results_df[columns],
                'donut_plot': donut_plot,
                'distribution_plot': probability_dist_plot
                }

        return results
    @staticmethod
    def run_evaluation(meta_df,
                        pair_id,
                        data_loader, 
                        probability_plot = True, 
                        metrics=True, 
                        confusion_matrix=True, 
                        basic_summary=True):
            
            # 1. Load model state
            device, model = PvDeepSet_model.load_model_state()

            # 2. Get probabilities
            actual_probs, predicted_probs = PvDeepSet_model.get_model_results(
                model=model, dataloader=data_loader, device=device, mode='evaluation'
            )

            class_names = ['C', 'L', 'I']

            # --- 3. CREATE CONSOLIDATED RESULTS TABLE ---
            eval_df = pd.DataFrame({
                'pair_id': pair_id,
                'true_C': actual_probs[:, 0],
                'true_L': actual_probs[:, 1],
                'true_I': actual_probs[:, 2],
                'pred_C': predicted_probs[:, 0],
                'pred_L': predicted_probs[:, 1],
                'pred_I': predicted_probs[:, 2]
            })

            eval_df['true_class'] = [class_names[i] for i in np.argmax(actual_probs, axis=1)]

            def classify_with_threshold(row):
                probs = [row['pred_C'], row['pred_L'], row['pred_I']]
                max_p = max(probs)
                if max_p < 0.5:
                    return "undefined"
                return class_names[np.argmax(probs)]

            eval_df['predicted_class'] = eval_df.apply(classify_with_threshold, axis=1)

            # --- 4. PREPARE DATA FOR PLOTS ---
            name_to_idx = {name: i for i, name in enumerate(class_names)}
            y_true_hard = np.array([name_to_idx[c] for c in eval_df['true_class']])
            y_pred_hard = np.array([name_to_idx.get(c, -1) for c in eval_df['predicted_class']])

            mask = y_pred_hard != -1
            y_true_filtered = y_true_hard[mask]
            y_pred_filtered = y_pred_hard[mask]
                
            # --- INNER FUNCTION 1: PROBABILITY REGRESSION ---
            def get_basic_summary(eval_df):
                return f"\n--- Prediction Summary --- {eval_df['predicted_class'].value_counts()} --------------------------\n"
            
            # --- INNER FUNCTION 2: PROBABILITY REGRESSION ---
            def get_prob_plot(y_true, y_prob, names):
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
                for i, ax in enumerate(axes):
                    actual, predicted = y_true[:, i], y_prob[:, i]
                    jitter = np.random.normal(0, 0.008, size=actual.shape)
                    sns.regplot(x=actual + jitter, y=predicted, ax=ax, 
                                scatter_kws={'alpha': 0.3, 's': 8, 'color': "#1672cf"}, 
                                line_kws={'color': '#e74c3c', 'linewidth': 2.5})
                    
                    ax.set_title(f"{names[i]}\n$R^2 = {r2_score(actual, predicted):.3f}$", pad=20, fontsize=15)
                    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
                plt.tight_layout()
                plt.close(fig)
                return fig

            # --- INNER FUNCTION 3: METRICS BAR & TABLE ---
            def get_metrics_plot(y_true, y_pred, names):
                report = classification_report(y_true, y_pred, target_names=names, output_dict=True)
                df = pd.DataFrame(report).transpose()
                
                fig = plt.figure(figsize=(8, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.5], hspace=0.3)
                ax_bar = fig.add_subplot(gs[0])
                ax_tbl = fig.add_subplot(gs[1])

                # Bar Chart
                df.loc[names, ['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax_bar, color=['#1672cf', '#4da1ff', '#aecff2'])
                ax_bar.set_title("Performance Metrics", fontsize=15, weight='bold')
                ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

                # Table
                ax_tbl.axis('off')
                the_table = ax_tbl.table(cellText=df.round(3).values, rowLabels=df.index, 
                                        colLabels=df.columns, loc='center', cellLoc='center')
                the_table.scale(1, 1.5)
                
                plt.close(fig)
                return fig

            # --- INNER FUNCTION 4: CONFUSION MATRIX ---
            def get_conf_matrix_plot(y_true, y_pred, names):
                cm = confusion_matrix(y_true, y_pred)
                cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                annot_labels = (np.array([f"{c}\n({p:.1%})" for c, p in zip(cm.flatten(), cm_perc.flatten())])).reshape(3, 3)
                
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(cm_perc, annot=annot_labels, fmt="", cmap="Blues", ax=ax,
                            xticklabels=names, yticklabels=names, annot_kws={"size": 12, "weight": "bold"})
                ax.set_title("Confusion Matrix", pad=20, fontsize=15, weight='bold')
                ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                
                plt.close(fig)
                return fig
            
            results_df = eval_df.merge(meta_df, left_on='pair_id', right_on='sample_id_paired', how='inner')

            # 5. Execute and Collect Plots
            results = {
                "results_table": results_df,
                "data": (actual_probs, predicted_probs, y_true_hard, y_pred_hard)
            }

            if basic_summary:
                results['basic_summary'] = get_basic_summary(eval_df)
            if probability_plot:
                results['prob_plot'] = get_prob_plot(actual_probs, predicted_probs, class_names)
            
            if metrics:
                results['metrics_plot'] = get_metrics_plot(y_true_filtered, y_pred_filtered, class_names)
                
            if confusion_matrix:
                results['conf_plot'] = get_conf_matrix_plot(y_true_filtered, y_pred_filtered, class_names)

            return results

