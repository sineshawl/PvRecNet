import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import json
import base64
import io
from services.api import PvRecNetAPI 

# 1. Initialize App
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    title="PvRecNet",
    suppress_callback_exceptions=True 
)

# 2. Define the Main App Layout (Navigation + Page Container)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Tracks the URL
    dcc.Store(id='dataset-id-store', storage_type='session'),
    html.Div(id='page-content')            # Where different pages will be rendered
])

# --- PAGE 1: THE HELP / GUIDELINE PAGE ---
# --- PAGE 1: THE HELP / GUIDELINE PAGE ---
def render_help_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                # Back Button
                dbc.Button("← Back to System", href="/", color="link", className="mt-4 p-0 text-decoration-none"),
                
                html.Div([
                    # Header Section
                    html.Div([
                        html.H1("System Guidelines & Documentation", className="display-5 text-primary"),
                        html.P([
                            "PvRecNet is a deep learning-based model for ", 
                            html.B(html.I("Plasmodium vivax")), # 1. Bold & Italic
                            " recurrence classification."
                        ], className="lead"),
                    ], className="mb-5"),

                    # 4. Visual Appeal: Use Cards for Sections
                    dbc.Row([
                        # Section: Description
                        dbc.Col([
                            html.H5("Model Description", className="text-info"),
                            html.P([
                                "Our architecture differentiates between ",
                                html.B("Recrudescence, Relapse, and Reinfection"),
                                ". It is trained on probabilistic outputs from ",
                                html.A("Pv3Rs", href="https://github.com/aimeertaylor/Pv3Rs", target="_blank"), # 2. Link
                                " to ensure robust genomic classification."
                            ]),
                        ], md=6),
                        
                        # Section: Data Requirements
                        dbc.Col([
                            html.H5("Data Requirements", className="text-info"),
                            html.P("Your dataset must contain these exact columns:"),
                            html.Div([
                                dbc.Badge("patient_id", color="primary", className="me-1"),
                                dbc.Badge("episode", color="primary", className="me-1"),
                                dbc.Badge("marker", color="primary", className="me-1"),
                                dbc.Badge("allele", color="primary", className="me-1"),
                                dbc.Badge("region", color="primary", className="me-1"),
                            ], className="mb-3"),
                            html.Small("Note: marker = loci, allele = haplotype, region = country.", className="text-muted")
                        ], md=6),
                    ], className="mb-4"),

                    html.Hr(),

                    # Section: Limitations
                    html.H4("⚠️ Scope & Limitations", className="mt-4 text-warning"),
                    html.Ul([
                        html.Li([html.B("Dependency on Pv3Rs: "), "Since the model is trained on the probabilistic outputs of Pv3Rs, its accuracy is bounded by Pv3Rs. Systematic errors in Pv3Rs may be replicated."]),
                        html.Li([html.B("Data Requirements: "), "Currently works only with amplicon sequencing datasets containing multiple markers and alleles."]),
                        html.Li([html.B("Episode Constraints: "), "Designed specifically for paired-episode classifications; it does not address scenarios with more than two episodes per individual as a single classification."]),
                        html.Li([html.B("Generalization: "), "The model may not generalize well to genetic markers or alleles unobserved during training."]),
                        html.Li([html.B("Protocol Dependent: "), "Trained specifically on ", html.A("pvAmpSeq", href="https://pubmed.ncbi.nlm.nih.gov/40492072/", target="_blank"), "; it only generalizes when using the same markers and allele names specified in that protocol."]),

                    ], className="text-secondary"),

                    # 4 & 5. Sample Dataset Section (Visual Highlight)
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Practice with Sample Data", className="card-title"),
                                    html.P("Download our demonstration file to see how the markers and episodes should be formatted."),
                                ], md=8),
                                dbc.Col([
                                    dbc.Button(
                                        "📥 Download Sample CSV", 
                                        id="download-sample-btn",
                                        color="success",
                                        className="w-100 mt-2"
                                    ),
                                    dcc.Download(id="download-sample-csv"),
                                ], md=4),
                            ])
                        ])
                    ], color="light", className="mt-5 border-start border-success border-4"),

                ], className="p-5 bg-white shadow-sm rounded-3 mb-5")
            ], md=10, className="offset-md-1")
        ])
    ], fluid=True)

# --- PAGE 2: THE MAIN SYSTEM (Tabs) ---
def render_main_app():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("PvRecNet", className="text-center my-4", style={"color":'#113d59'}), width=10),
            # Help Link in Top Right
            dbc.Col(html.A("Help & Guidelines", href="/help", className="text-muted", style={"marginTop": "30px", "display": "block"}), width=2)
        ]),

        dbc.Tabs(id="main-tabs", active_tab="tab-intro", children=[
            # Tab 0: Welcome / Landing
            dbc.Tab(label="Welcome", tab_id="tab-intro", active_label_style={"color": '#113d59', "fontWeight": "bold"}, children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H2("Welcome to PvRecNet", className="display-5 mt-5", style={"color":'#113d59'}),
                                html.P("Deep Learning support for Malaria Recurrence Classification.", className="lead"),
                                html.P([
                                    "Please see our ", 
                                    html.A("Help Page", href="/help", style={"fontWeight": "bold"}),
                                    " to understand exactly how the system works and how to format your data."
                                ]),
                                html.Hr(),
                                dbc.Button("Get Started →", id="get-started-btn", color="primary", size="lg", className="mt-3 shadow")
                            ], className="p-5 bg-light border rounded-3 text-center")
                        ], md=8, className="offset-md-2")
                    ])
                ], fluid=True)
            ]),

            # Tab 1: Prediction Workflow
            dbc.Tab(label="Prediction Workflow", tab_id="tab-predict", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.B("Step 1: Upload Data")),
                            dbc.CardBody([
                                dcc.Upload(id='upload-data', children=html.Div(['Drag/Drop or Select CSV']),
                                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'}),
                                html.Div(id='upload-status', className="mt-2"),
                                html.Hr(),
                                dbc.Button("Run Deep Sets Prediction", id="predict-btn", color="primary", disabled=True, className="w-100")
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.B("Step 2: Data Workbench")),
                            dbc.CardBody(dcc.Loading(dash_table.DataTable(
                                id='preview-table', columns=[], data=[], export_format='csv', editable=True, 
                                filter_action="native", sort_action="native", page_size=10,
                                style_table={'height': '350px', 'overflowY': 'auto'},
                                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                style_cell={'textAlign': 'left', 'padding': '10px'}
                            )))
                        ])
                    ], md=8)
                ], className="mt-4"),
                dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader(html.B("Step 3: Model Results")), dbc.CardBody(dcc.Loading(id="prediction-output"))], className="mt-4 mb-5"), width=12))
            ]),

            # # Tab 2: Analytics Tools
            # dbc.Tab(label="Analytics Tools", tab_id="tab-analytics", children=[
            #     dbc.Row(dbc.Col(dbc.Card([
            #         dbc.CardHeader(html.B("Genomic Analytics")),
            #         dbc.CardBody([
            #             dbc.Button("Compute MOI", id="moi-btn", color="info", className="me-2"),
            #             dbc.Button("Compute Allele Freq", id="freq-btn", color="secondary"),
            #             html.Hr(), dcc.Loading(id="analytics-output")
            #         ])
            #     ], className="mt-4"), md=12))
            # ])
        ])
    ], fluid=True)

@app.callback(
    Output("download-sample-csv", "data"),
    Input("download-sample-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_data(n_clicks):
    # Create a small dummy dataframe that matches your 5 required columns
    sample_df = pd.read_csv('../backend/PvDeepSetNet/datasets/GUI_sample_dataset.csv')
    return dcc.send_data_frame(sample_df.to_csv, "pvrecnet_sample_data.csv", index=False)

# --- ROUTER CALLBACK ---
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/help':
        return render_help_page()
    else:
        return render_main_app()

# --- NAVIGATION CALLBACK ---
@app.callback(
    Output("main-tabs", "active_tab"),
    Input("get-started-btn", "n_clicks"),
    prevent_initial_call=True
)
def navigate_to_predict(n_clicks):
    return "tab-predict" if n_clicks else "tab-intro"

# --- FUNCTIONAL CALLBACKS (Upload, Predict, Analytics) ---

@app.callback(
    [Output('dataset-id-store', 'data'), Output('upload-status', 'children'), Output('predict-btn', 'disabled'), 
     Output('preview-table', 'data'), Output('preview-table', 'columns'), Output('preview-table', 'fixed_rows')],
    [Input('upload-data', 'contents')], [State('upload-data', 'filename')], prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return dash.no_update
    try:
        decoded = base64.b64decode(contents.split(',')[1])
        df = pd.read_csv(io.BytesIO(decoded))
        cols = [{"name": i, "id": i, "editable": True, "type": 'numeric' if pd.api.types.is_numeric_dtype(df[i]) else 'text'} for i in df.columns]
        res = PvRecNetAPI.upload_dataset(decoded, filename)
        return res.get("dataset_id"), dbc.Alert(f"Ready: {filename}", color="success"), False, df.to_dict('records'), cols, {'headers': True}
    except Exception as e:
        return None, dbc.Alert(f"Error: {str(e)}", color="danger"), True, [], [], {'headers': False}

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('dataset-id-store', 'data'),
    prevent_initial_call=True
)

def run_predict(n_clicks, dataset_id):
    try:
        res = PvRecNetAPI.run_prediction(dataset_id)
        fig1 = json.loads(res['plot_json1'])
        fig2 = json.loads(res['plot_json2'])

        # 1. Convert the results list to a DataFrame
        df = pd.DataFrame(res['results_table'])

        # 2. Identify the probability columns (adjust names based on your actual API output)
        prob_cols = ['C_marg', 'L_marg', 'I_marg'] # or 'post_C', etc.
        
        # 3. Apply string formatting for consistent visual alignment
        for col in prob_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

        # 4. Create the DataTable using the formatted data
        table = dash_table.DataTable(
            data=df.to_dict('records'), 
            columns=[{"name": i, "id": i} for i in df.columns], 
            export_format='csv', 
            page_size=10, 
            style_table={'height': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left'} # Optional: improves readability
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)], md=5), 
                dbc.Col(table, md=7)
            ])
        ])
    except Exception as e: 
        return dbc.Alert(f"Error: {str(e)}", color="danger")

# @app.callback(
#     Output('analytics-output', 'children'),
#     [Input('moi-btn', 'n_clicks'), Input('freq-btn', 'n_clicks')],
#     State('dataset-id-store', 'data'),
#     prevent_initial_call=True
# )
# def handle_analytics(moi_n, freq_n, dataset_id):
#     ctx = callback_context
#     bid = ctx.triggered[0]['prop_id'].split('.')[0]
#     try:
#         if bid == "moi-btn":
#             res = PvRecNetAPI.get_moi(dataset_id)
#             return dash_table.DataTable(data=res, columns=[{"name": i, "id": i} for i in pd.DataFrame(res).columns], export_format='csv')
#         else:
#             res = PvRecNetAPI.get_allele_freq(dataset_id)
#             import plotly.express as px
#             return dcc.Graph(figure=px.pie(res, names='allele', values='frequency'))
#     except Exception as e: return dbc.Alert(str(e), color="danger")

if __name__ == '__main__':
    app.run(debug=True, port=8050)