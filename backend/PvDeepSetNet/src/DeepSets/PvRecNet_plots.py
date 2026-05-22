import pandas as pd
import plotly.express as px

class PvRecNet_Plots:
    @staticmethod
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

    @staticmethod
    def plot_probability_dist(df, cols=['sample_id', 'prob_C', 'prob_L','prob_I','predicted_class']):

        df = df.copy()
        df = df[cols].drop_duplicates().reset_index(drop=True)

        # Grouped and sorted blocks
        df_C = df[df.predicted_class == 'C'].sort_values(cols[1], ascending=False).reset_index(drop=True)
        df_L = df[df.predicted_class == 'L'].sort_values(cols[2], ascending=False).reset_index(drop=True)
        df_I = df[df.predicted_class == 'I'].sort_values(cols[3], ascending=False).reset_index(drop=True)

        df_sorted = pd.concat([df_C, df_L, df_I], ignore_index=True)

        df_long = df_sorted.melt(
            id_vars=cols[0], 
            value_vars=cols[1:4],
            var_name='Probability_Type', 
            value_name='Probability'
        )
        
        fig = px.bar(
            df_long, 
            x=cols[0], 
            y="Probability", 
            color="Probability_Type",
            color_discrete_sequence=['#ff4d00', '#e6b400', '#1fb3c4'],
            category_orders={cols[0]: df_sorted[cols[0]].tolist()},
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