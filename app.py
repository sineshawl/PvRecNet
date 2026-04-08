import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
from services.api import PvRecNetAPI 

# Initialize App with a professional Bootstrap theme (CYBORG or FLATLY)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dcc.Store(id='dataset-id-store'),
    
    dbc.Row(dbc.Col(html.H1("PvRecNet", className="text-center my-4"))),

    dbc.Tabs([
        # TAB 1: UPLOAD & PREDICT
        dbc.Tab(label="Prediction Workflow", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Step 1: Upload Data"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div(['Drag/Drop or ', html.A('Select CSV')]),
                                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                       'borderWidth': '1px', 'borderStyle': 'dashed', 'textAlign': 'center'}
                            ),
                            html.Div(id='upload-status', className="mt-2"),
                            dbc.Button("Run Prediction", id="predict-btn", color="primary", 
                                       disabled=True, className="mt-3 w-100")
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Step 2: Results"),
                        dbc.CardBody(dcc.Loading(id="prediction-output"))
                    ])
                ], md=8)
            ], className="mt-4")
        ]),

        # TAB 2: ANALYTICS
        dbc.Tab(label="Analytics Tools", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Button("Compute MOI", id="moi-btn", color="info", className="m-2"),
                    dbc.Button("Compute Allele Freq", id="freq-btn", color="secondary", className="m-2"),
                    html.Hr(),
                    dcc.Loading(id="analytics-output")
                ], md=12)
            ], className="mt-4")
        ])
    ])
], fluid=True)

# --- Callbacks ---

@app.callback(
    Output('dataset-id-store', 'data'),
    Output('upload-status', 'children'),
    Output('predict-btn', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if not contents: return dash.no_update
    decoded = base64.b64decode(contents.split(',')[1])
    try:
        res = PvRecNetAPI.upload_dataset(decoded, filename)
        return res['dataset_id'], dbc.Alert(f"Ready: {filename}", color="success"), False
    except Exception as e:
        return None, dbc.Alert(f"Error: {e}", color="danger"), True

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('dataset-id-store', 'data'),
    prevent_initial_call=True
)
def run_predict(n, ds_id):
    try:
        res = PvRecNetAPI.run_prediction(ds_id)
        fig = px.bar(x=list(res['probabilities'].keys()), y=list(res['probabilities'].values()), title="Probability")
        return html.Div([
            html.H3(f"Prediction: {res['prediction']}"),
            dcc.Graph(figure=fig)
        ])
    except Exception as e: return dbc.Alert(f"Error: {e}", color="danger")

@app.callback(
    Output('analytics-output', 'children'),
    Input('moi-btn', 'n_clicks'),
    Input('freq-btn', 'n_clicks'),
    State('dataset-id-store', 'data'),
    prevent_initial_call=True
)
def run_analytics(m_n, f_n, ds_id):
    if not ds_id: return dbc.Alert("No dataset found.", color="warning")
    trigger = dash.callback_context.triggered[0]['prop_id']
    try:
        if "moi-btn" in trigger:
            res = PvRecNetAPI.get_moi(ds_id)
            return dash_table.DataTable(res, page_size=10)
        else:
            res = PvRecNetAPI.get_allele_freq(ds_id)
            return dcc.Graph(figure=px.pie(res, names='allele', values='frequency'))
    except Exception as e: return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)