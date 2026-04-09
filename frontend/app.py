import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import io

# Import your custom API service
from services.api import PvRecNetAPI 

# 1. Initialize the Dash App
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    title="PvRecNet Malaria System"
)

# 2. Define the Layout
app.layout = dbc.Container([
    # Browser-side storage for the dataset ID
    dcc.Store(id='dataset-id-store', storage_type='session'),
    
    # Header
    dbc.Row(dbc.Col(html.H1("PvRecNet: Malaria Recurrence Classifier", className="text-center my-4"), width=12)),

    dbc.Tabs([
        # --- TAB 1: PREDICTION WORKFLOW ---
        dbc.Tab(label="Prediction Workflow", children=[
            dbc.Row([
                # Left Column: Controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Step 1: Upload & Action")),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div(['Drag/Drop or ', html.A('Select CSV')]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed', 
                                    'borderRadius': '5px', 'textAlign': 'center'
                                }
                            ),
                            html.Div(id='upload-status', className="mt-2"),
                            html.Hr(),
                            dbc.Button(
                                "Run Deep Sets Prediction", 
                                id="predict-btn", 
                                color="primary", 
                                disabled=True, 
                                className="w-100"
                            ),
                            html.Small("Note: Prediction uses the data currently in the table.", 
                                       className="text-muted mt-2 d-block")
                        ])
                    ])
                ], md=4),

                # Right Column: Interactive Data Preview
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Step 2: Interactive Data Workbench")),
                        dbc.CardBody([
                            dcc.Loading(id="loading-preview", children=[
                                dash_table.DataTable(
                                    id='preview-table',
                                    columns=[],  # <--- CRITICAL: Initialize as empty list
                                    data=[],     # <--- CRITICAL: Initialize as empty list
                                    
                                    # Interactivity
                                    editable=True,
                                    # filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    row_deletable=True,
                                    
                                    # Pagination & Scrolling
                                    page_action="native",
                                    page_current=0,
                                    page_size=15,
                                    fixed_rows={'headers': True},
                                    
                                    style_table={
                                        'height': '450px',
                                        'overflowY': 'auto',
                                        'overflowX': 'auto',
                                    },
                                    
                                    # Visual Styling
                                    style_header={
                                        'backgroundColor': '#f8f9fa',
                                        'fontWeight': 'bold',
                                        'border': '1px solid #dee2e6'
                                    },
                                    style_cell={
                                        'textAlign': 'left',
                                        'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                                        'minWidth': '120px', 'width': '150px', 'maxWidth': '300px',
                                        'padding': '8px'
                                    },
                                    style_data_conditional=[
                                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(250, 250, 250)'},
                                        {'if': {'state': 'active'}, 'backgroundColor': 'rgba(0, 123, 255, 0.1)', 'border': '1px solid #007bff'}
                                    ]
                                )
                            ])
                        ])
                    ])
                ], md=8)
            ], className="mt-4"),

            # Bottom Row: Inference Results
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Step 3: Model Output")),
                        dbc.CardBody(dcc.Loading(id="prediction-output"))
                    ], className="mt-4 mb-5")
                ], width=12)
            ])
        ]),

        # --- TAB 2: ANALYTICS TOOLS ---
        dbc.Tab(label="Analytics Tools", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Genomic Analytics")),
                        dbc.CardBody([
                            html.P("These tools operate on the uploaded dataset ID independently of the classifier."),
                            dbc.Button("Compute MOI", id="moi-btn", color="info", className="me-2"),
                            dbc.Button("Compute Allele Freq", id="freq-btn", color="secondary"),
                            html.Hr(),
                            dcc.Loading(id="analytics-output")
                        ])
                    ])
                ], md=12)
            ], className="mt-4")
        ])
    ])
], fluid=True)

# --- CALLBACKS ---

# 1. Handle File Upload, API Registration, and Table Population
@app.callback(
    Output('dataset-id-store', 'data'),
    Output('upload-status', 'children'),
    Output('predict-btn', 'disabled'),
    Output('preview-table', 'data'),
    Output('preview-table', 'columns'),
    Output('preview-table', 'fixed_rows'), # NEW: Dynamic fixed rows
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload_and_preview(contents, filename):
    if not contents:
        return dash.no_update
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.BytesIO(decoded))
        
        preview_cols = []
        for i in df.columns:
            column_type = 'numeric' if pd.api.types.is_numeric_dtype(df[i]) else 'text'
            preview_cols.append({"name": i, "id": i, "editable": True, "type": column_type})

        api_response = PvRecNetAPI.upload_dataset(decoded, filename)
        dataset_id = api_response.get("dataset_id")
        
        # We now return {'headers': True} as the last argument
        return dataset_id, dbc.Alert(f"Loaded {filename}", color="success"), False, \
               df.head(200).to_dict('records'), preview_cols, {'headers': True}

    except Exception as e:
        # Return {'headers': False} on error to keep it from crashing
        return None, dbc.Alert(f"Error: {str(e)}", color="danger"), True, [], [], {'headers': False}

# 2. Run Model Prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('dataset-id-store', 'data'),
    prevent_initial_call=True
)
def run_model_inference(n_clicks, dataset_id):
    if not dataset_id:
        return dbc.Alert("No dataset found. Please upload a file first.", color="warning")
    
    try:
        # Call Backend API
        res = PvRecNetAPI.run_prediction(dataset_id)
        
        # Plotly Bar Chart for Probability
        probs = res.get('probabilities', {})
        fig = px.bar(
            x=list(probs.keys()), 
            y=list(probs.values()),
            labels={'x': 'Class', 'y': 'Probability'},
            title="Deep Sets Probability Distribution",
            color=list(probs.keys()),
            template="plotly_white"
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Classification Result"),
                        html.H2(res['prediction'], className="text-primary"),
                        html.P(f"Model Confidence: {res['confidence']:.2%}")
                    ], className="p-3 border rounded bg-light")
                ], md=4),
                dbc.Col([
                    dcc.Graph(figure=fig, style={'height': '300px'})
                ], md=8)
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"Inference Failed: {str(e)}", color="danger")

# 3. Analytics Tools (MOI & Allele Frequency)
@app.callback(
    Output('analytics-output', 'children'),
    Input('moi-btn', 'n_clicks'),
    Input('freq-btn', 'n_clicks'),
    State('dataset-id-store', 'data'),
    prevent_initial_call=True
)
def handle_analytics(moi_n, freq_n, dataset_id):
    if not dataset_id:
        return dbc.Alert("No data available for analytics. Go to Prediction tab to upload.", color="warning")
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if button_id == "moi-btn":
            res = PvRecNetAPI.get_moi(dataset_id)
            return html.Div([
                html.H5("Multiplicity of Infection (MOI) Table"),
                dash_table.DataTable(
                    data=res, 
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                    style_table={'overflowX': 'auto'}
                )
            ])
        
        elif button_id == "freq-btn":
            res = PvRecNetAPI.get_allele_freq(dataset_id)
            # Assuming res is a list of {'allele': str, 'frequency': float}
            fig = px.pie(res, names='allele', values='frequency', title="Allele Frequency Distribution")
            return dcc.Graph(figure=fig)
            
    except Exception as e:
        return dbc.Alert(f"Analytical Tool Error: {str(e)}", color="danger")

# 4. Main Entry Point
if __name__ == '__main__':
    # Using app.run() for modern Dash compatibility
    app.run(debug=True, port=8050)