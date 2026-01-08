import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, no_update
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# must have train.csv locally to run
# https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data?select=test.zip


# Data Preprocessing
SAMPLE_SIZE = 10000 


def load_data():
    # 1. Construct the absolute path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'train.csv')
    
    print(f"DEBUG: Attempting to read: {file_path}")

    try:
        # 2. Check if file physically exists before reading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        # 3. Load Data
        df = pd.read_csv(file_path, nrows=10000)
        print(f"SUCCESS: Loaded {len(df)} rows.")

        if 'store_and_fwd_flag' in df.columns:
            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
        
        df['month'] = df['pickup_datetime'].dt.month
        df['day_of_week'] = df['pickup_datetime'].dt.day_name()
        df['hour_of_day'] = df['pickup_datetime'].dt.hour
        df['day_of_year'] = df['pickup_datetime'].dt.dayofyear
        
        # Vectorized Haversine
        def haversine_vectorized(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return c * 6371
            
        df['distance_km'] = haversine_vectorized(
            df['pickup_longitude'], df['pickup_latitude'],
            df['dropoff_longitude'], df['dropoff_latitude']
        )
        
        # Pre Distance Filter and Norm
        df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 14400)]
        df['log_trip_duration'] = np.log1p(df['trip_duration'])

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days_order, ordered=True)

        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42)
            
        return df

    except Exception as e:
        print(f"\nCRITICAL ERROR LOADING DATA: {e}\n")
        return pd.DataFrame({'trip_duration': [100, 200, 100000]}) # Fallback

raw_df = load_data()
numerical_cols = ['trip_duration', 'log_trip_duration', 'distance_km', 'passenger_count', 'hour_of_day', 'pickup_longitude', 'pickup_latitude']
categorical_cols = ['day_of_week', 'vendor_id', 'month', 'store_and_fwd_flag']
all_cols = raw_df.columns.tolist()


# App Setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="NYC Taxi Dashboard", suppress_callback_exceptions=True)

tab_style = {'padding': '10px', 'fontWeight': 'bold'}
selected_style = {'padding': '10px', 'borderTop': '3px solid #119DFF', 'backgroundColor': '#f9f9f9', 'fontWeight': 'bold'}
control_box_style = {'padding': '20px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px', 'marginBottom': '20px'}

# App Layout
app.layout = html.Div([
    dcc.Store(id='stored-data', data=raw_df.to_dict('records')),

    # A. Header
    html.Div([
        html.Img(src='assets/taxi_clip.svg', style={'height':'40px', 'float':'left', 'marginRight':'15px'}),
        html.H1("NYC Taxi Dashboard", style={'marginBottom': '0px'}),
        html.H5("Interactive Data Pipeline", style={'color': '#555', 'marginTop': '0px'}),
        html.Hr()
    ]),

    # B. Main Tabs
    dcc.Tabs(id="main-tabs", value='tab-cleaning', children=[
        
        # 1. Data Cleaning
        dcc.Tab(label='1. Data Cleaning', value='tab-cleaning', style=tab_style, selected_style=selected_style, children=[
            html.Div([
                html.H3("Data Cleaning & Outliers"),
                html.Div([
                    html.Label("Step 1: Cleaning Operations"),
                    dcc.Checklist(id='cleaning-checklist', options=[{'label': ' Drop Duplicates', 'value': 'duplicates'}, {'label': ' Drop Null Values', 'value': 'nulls'}, {'label': ' Filter Negative Durations', 'value': 'negatives'}], value=[], inline=True),
                    html.Br(),
                    html.Label("Step 2: Outlier Removal (Z-Score)"),
                    dcc.Dropdown(
                        id='outlier-method', 
                        options=[
                            {'label': 'Keep Outliers (None)', 'value': 'none'},
                            {'label': 'Remove Extreme (Z > 3)', 'value': 'z3'},
                            {'label': 'Remove Moderate (Z > 2)', 'value': 'z2'},
                            {'label': 'Remove Outliers (IQR Method)', 'value': 'iqr'}
                        ], 
                        value='none', 
                        clearable=False
                    ),
                    html.Br(),
                    html.Button("Apply & Download CSV", id="btn-download", className="button-primary"),
                    dcc.Download(id="download-csv"),
                    html.Div(id='cleaning-stats', style={'marginTop': '10px', 'fontWeight': 'bold', 'color': '#0074D9'})
                ], style=control_box_style),
                html.H5("Dataset Preview"),
                dash_table.DataTable(id='data-preview', page_size=10, style_table={'overflowX': 'auto'})
            ], style={'padding': '20px'})
        ]),

        # 2. Transformation
        dcc.Tab(label='2. Transform & Norm', value='tab-transform', style=tab_style, selected_style=selected_style, children=[
            html.Div([
                html.H3("Feature Transformation"),
                html.Div([
                    html.Div([
                        html.Label("Feature:"), dcc.Dropdown(id='transform-col', options=[{'label': c, 'value': c} for c in numerical_cols], value='trip_duration'),
                        html.Br(), html.Label("Transform:"), dcc.RadioItems(id='transform-type', options=[{'label': ' None', 'value': 'none'}, {'label': ' Log', 'value': 'log'}, {'label': ' Sqrt', 'value': 'sqrt'}, {'label': ' Box-Cox', 'value': 'boxcox'}], value='none', labelStyle={'display': 'block'})
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([dcc.Loading(dcc.Graph(id='transform-graph')), html.Div(id='normality-result')], style={'width': '65%', 'display': 'inline-block'})
                ])
            ], style={'padding': '20px'})
        ]),

        # 3. PCA
        dcc.Tab(label='3. PCA Analysis', value='tab-pca', style=tab_style, selected_style=selected_style, children=[
            html.Div([
                html.H3("PCA Analysis"),
                html.Div([html.Label("Components:"), dcc.Slider(id='pca-slider', min=2, max=len(numerical_cols), step=1, value=2, marks={i: str(i) for i in range(2, len(numerical_cols)+1)})], style=control_box_style),
                dcc.Graph(id='pca-graph')
            ], style={'padding': '20px'})
        ]),

        # 4. Visualizations
        dcc.Tab(label='4. Interactive Visualizations', value='tab-viz', style=tab_style, selected_style=selected_style, children=[
            html.Div([
                html.H3("Exploratory Data Analysis (EDA)"),
                
                # 1. ANALYSIS GOAL SELECTOR
                html.Div([
                    html.Label("Select Analysis Goal:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                    dcc.RadioItems(
                        id='analysis-goal',
                        options=[
                            {'label': ' 1. Numerical Distributions (Univariate)', 'value': 'group-1'},
                            {'label': ' 2. Categorical Counts (Univariate)', 'value': 'group-2'},
                            {'label': ' 3. Numerical Relationships (Bivariate)', 'value': 'group-3'},
                            {'label': ' 4. Categorical Comparison (Cat vs Num)', 'value': 'group-4'},
                            {'label': ' 5. Global Matrix (Multivariate)', 'value': 'group-5'}
                        ],
                        value='group-1',
                        labelStyle={'display': 'inline-block', 'marginRight': '20px', 'marginTop': '10px'}
                    )
                ], style={'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '5px', 'marginBottom': '20px'}),

                html.Div([
                    # 2. LEFT: DYNAMIC CONTROLS
                    html.Div([
                        html.H5("Controls"),
                        html.Hr(),
                        
                        # GROUP 1 CONTROLS (Numerical Dist)
                        html.Div(id='controls-group-1', children=[
                            html.Label("Select Numerical Feature:"),
                            dcc.Dropdown(
                                id='viz-num-feature',
                                options=[{'label': c, 'value': c} for c in numerical_cols],
                                value='log_trip_duration',
                                clearable=False
                            ),
                            html.Br(),
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(
                                id='viz-plot-type-g1',
                                options=[
                                    {'label': 'Dist Plot (Histogram)', 'value': 'dist'},
                                    {'label': 'KDE Plot (Filled)', 'value': 'kde'},
                                    {'label': 'Rug Plot', 'value': 'rug'},
                                    {'label': 'QQ Plot (Normality)', 'value': 'qq'}
                                ],
                                value='dist',
                                clearable=False
                            ),
                            html.Br(),
                            html.Div(id='stats-card-g1', style={'padding': '15px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                        ], style={'display': 'block'}), 

                        # GROUP 2 CONTROLS (Categorical Counts)
                        html.Div(id='controls-group-2', children=[
                            html.Label("Select Categorical Feature:"),
                            dcc.Dropdown(
                                id='viz-cat-feature',
                                options=[{'label': c, 'value': c} for c in categorical_cols],
                                value='day_of_week',
                                clearable=False
                            ),
                            html.Br(),
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(
                                id='viz-plot-type-g2',
                                options=[
                                    {'label': 'Count Plot (Bar)', 'value': 'count'},
                                    {'label': 'Pie Chart', 'value': 'pie'}
                                ],
                                value='count',
                                clearable=False
                            ),
                            html.Br(),
                            html.Div(id='stats-card-g2', style={'padding': '15px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                        ], style={'display': 'none'}),

                        # GROUP 3 CONTROLS (Numerical Relationships)
                        html.Div(id='controls-group-3', children=[
                            html.Label("Select X-Axis (Num):"),
                            dcc.Dropdown(id='viz-x-g3', options=[{'label': c, 'value': c} for c in numerical_cols], value='distance_km', clearable=False),
                            html.Br(),
                            html.Label("Select Y-Axis (Num):"),
                            dcc.Dropdown(id='viz-y-g3', options=[{'label': c, 'value': c} for c in numerical_cols], value='trip_duration', clearable=False),
                            html.Br(),
                            html.Label("Select Color (Optional):"),
                            dcc.Dropdown(id='viz-color-g3', options=[{'label': c, 'value': c} for c in all_cols], value='vendor_id'),
                            html.Br(),
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(id='viz-plot-type-g3', options=[
                                {'label': 'Reg Plot (Scatter+Line)', 'value': 'reg'},
                                {'label': 'Joint Plot (Marginals)', 'value': 'joint'},
                                {'label': 'Hexbin Plot', 'value': 'hex'},
                                {'label': 'Contour Plot', 'value': 'contour'},
                                {'label': '3D Plot', 'value': '3d'},
                                {'label': 'Line Plot', 'value': 'line'},
                                {'label': 'Area Plot', 'value': 'area'}
                            ], value='reg', clearable=False),
                            html.Br(),
                            html.Div(id='stats-card-g3', style={'padding': '15px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                        ], style={'display': 'none'}),

                        # GROUP 4 CONTROLS (Categorical Comparison)
                        html.Div(id='controls-group-4', children=[
                            html.Label("Select X-Axis (Categorical):"),
                            dcc.Dropdown(id='viz-x-g4', options=[{'label': c, 'value': c} for c in categorical_cols], value='day_of_week', clearable=False),
                            html.Br(),
                            html.Label("Select Y-Axis (Numerical):"),
                            dcc.Dropdown(id='viz-y-g4', options=[{'label': c, 'value': c} for c in numerical_cols], value='trip_duration', clearable=False),
                            html.Br(),
                            html.Label("Select Color (Optional):"),
                            dcc.Dropdown(id='viz-color-g4', options=[{'label': c, 'value': c} for c in categorical_cols], value='vendor_id'),
                            html.Br(),
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(id='viz-plot-type-g4', options=[
                                {'label': 'Bar Plot (Mean)', 'value': 'bar'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Violin Plot', 'value': 'violin'},
                                {'label': 'Multivariate Boxen (Violin-Style)', 'value': 'boxen'},
                                {'label': 'Strip Plot', 'value': 'strip'},
                                {'label': 'Swarm Plot', 'value': 'swarm'}
                            ], value='bar', clearable=False),
                            html.Br(),
                            html.Label("Bar Mode (for Bar Plot):"),
                            dcc.RadioItems(id='viz-barmode-g4', options=[
                                {'label': ' Group', 'value': 'group'}, 
                                {'label': ' Stack', 'value': 'stack'}
                            ], value='group', inline=True),
                            html.Br(),
                            html.Div(id='stats-card-g4', style={'padding': '15px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                        ], style={'display': 'none'}),

                        # GROUP 5 CONTROLS (Global Matrix)
                        html.Div(id='controls-group-5', children=[
                            html.Label("Select Plot Type:"),
                            dcc.Dropdown(id='viz-plot-type-g5', options=[
                                {'label': 'Heatmap (Correlation)', 'value': 'heatmap'},
                                {'label': 'Pair Plot (Scatter Matrix)', 'value': 'pair'}
                            ], value='heatmap', clearable=False),
                            html.Br(),
                            html.Div(id='stats-card-g5', style={'padding': '15px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                        ], style={'display': 'none'}),

                    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'}),

                    # 3. RIGHT: GRAPH AREA
                    html.Div([
                        dcc.Loading(dcc.Graph(id='main-viz-graph', style={'height': '600px'}))
                    ], style={'width': '70%', 'display': 'inline-block', 'paddingLeft': '20px'})
                ])
            ], style={'padding': '20px'})
        ]),

        # 5. Formal Normality Testing Tab
        dcc.Tab(label='5. Hypothesis Testing', value='tab-hyp', style=tab_style, selected_style=selected_style, children=[
            html.Div([
                html.H3("Formal Normality Hypothesis Testing"),
                html.Div([
                    html.Div([
                        html.Label("Feature for Analysis:"),
                        dcc.Dropdown(id='hyp-col', options=[{'label': c, 'value': c} for c in numerical_cols], value='trip_duration'),
                        html.Br(),
                        html.Label("Filter by Vendor (Group):"),
                        dcc.RadioItems(id='hyp-vendor', options=[{'label': 'All', 'value': 'all'}, {'label': 'Vendor 1', 'value': 1}, {'label': 'Vendor 2', 'value': 2}], value='all', inline=True),
                        html.Br(),
                        html.Label("Select Test Method:"),
                        dcc.Dropdown(
                            id='hyp-test-method',
                            options=[
                                {'label': 'Kolmogorov-Smirnov (K-S)', 'value': 'ks'},
                                {'label': 'Shapiro-Wilk', 'value': 'shapiro'},
                                {'label': "D'Agostino K-Squared", 'value': 'k2'}
                            ],
                            value='ks',
                            clearable=False
                        ),
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
                    html.Div([
                        html.Div(id='hyp-test-results', style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': 'white'})
                    ], style={'width': '65%', 'display': 'inline-block', 'paddingLeft': '20px'})
                ])
            ], style={'padding': '20px'})
        ])
    ]),

    # C. Footer
    html.Div([
        html.Hr(),
        html.Label("User Observation Notes:"),
        dcc.Textarea(id='user-notes', placeholder='Type your findings here...', style={'width': '100%', 'height': 80})
    ], style={'padding': '20px', 'backgroundColor': '#fafafa'})
])

# Callbacks

# Tab 1: Cleaning & Download
@app.callback(
    [Output('stored-data', 'data'), Output('cleaning-stats', 'children'), Output('data-preview', 'data'), Output('download-csv', 'data')],
    [Input('cleaning-checklist', 'value'), Input('outlier-method', 'value'), Input('btn-download', 'n_clicks')]
)
def update_cleaning(cleaning_opts, outlier_method, n_clicks):
    df = raw_df.copy()
    orig = len(df)
    if cleaning_opts and 'duplicates' in cleaning_opts: df.drop_duplicates(inplace=True)
    if cleaning_opts and 'nulls' in cleaning_opts: df.dropna(inplace=True)
    if cleaning_opts and 'negatives' in cleaning_opts: df = df[df['trip_duration'] > 0]
    
    # 2. Outlier Removal (With safety check)
    if outlier_method != 'none':
        try:
            # Ensure no NaNs
            valid_rows = df['trip_duration'].notna()
            series = df.loc[valid_rows, 'trip_duration']
            
            if outlier_method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df.loc[valid_rows][(series >= lower) & (series <= upper)]
            
            else:
                z = np.abs(stats.zscore(series))
                thresh = 3 if outlier_method == 'z3' else 2
                df = df.loc[valid_rows][z < thresh]
        except:
            pass
        
    msg = f"Original: {orig} | Cleaned: {len(df)} | Dropped: {orig - len(df)}"
    
    dl = None
    if ctx.triggered and 'btn-download' in ctx.triggered[0]['prop_id']:
        dl = dcc.send_data_frame(df.to_csv, "processed_data.csv")
    return df.to_dict('records'), msg, df.head(10).to_dict('records'), dl

# Tab 2: Transform
@app.callback([Output('transform-graph', 'figure'), Output('normality-result', 'children')], [Input('transform-col', 'value'), Input('transform-type', 'value'), Input('stored-data', 'data')])
def update_transform(col, method, data):
    df = pd.DataFrame(data)
    vals = df[col]
    if method == 'log': vals = np.log1p(vals[vals>0])
    elif method == 'sqrt': vals = np.sqrt(vals[vals>=0])
    elif method == 'boxcox': vals = vals if (vals<=0).any() else stats.boxcox(vals)[0]
    
    fig = px.histogram(x=vals, title=f"Distribution: {col} ({method})")
    try:
        s, p = stats.shapiro(vals[:100])
        res = [html.B("Shapiro-Wilk:"), html.P(f"P={p:.4f} (" + ("Normal" if p>0.05 else "Not Normal") + ")")]
    except: res = "Test Failed"
    return fig, res

# Tab 3: PCA
@app.callback(Output('pca-graph', 'figure'), [Input('pca-slider', 'value'), Input('stored-data', 'data')])
def update_pca(n, data):
    # Safety Check: PCA Default
    if n is None: n = 2
    if data is None: return go.Figure().update_layout(title="Data Loading...")

    try:
        df = pd.DataFrame(data)
        X = df[numerical_cols].dropna()
        pca = PCA(n_components=n).fit(StandardScaler().fit_transform(X))
        fig = px.area(x=range(1, n+1), y=np.cumsum(pca.explained_variance_ratio_), title="PCA Explained Variance")
        fig.add_hline(y=0.95, line_dash="dot")
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"Error in PCA: {str(e)}")

# Tab 4: Visualizations
@app.callback(
    [Output('controls-group-1', 'style'), Output('controls-group-2', 'style'), 
     Output('controls-group-3', 'style'), Output('controls-group-4', 'style'),
     Output('controls-group-5', 'style')],
    [Input('analysis-goal', 'value')]
)
def toggle_controls(goal):
    hidden = {'display': 'none'}
    visible = {'display': 'block'}
    if goal == 'group-1': return visible, hidden, hidden, hidden, hidden
    elif goal == 'group-2': return hidden, visible, hidden, hidden, hidden
    elif goal == 'group-3': return hidden, hidden, visible, hidden, hidden
    elif goal == 'group-4': return hidden, hidden, hidden, visible, hidden
    elif goal == 'group-5': return hidden, hidden, hidden, hidden, visible
    return hidden, hidden, hidden, hidden, hidden

@app.callback(
    [Output('main-viz-graph', 'figure'), 
     Output('stats-card-g1', 'children'),
     Output('stats-card-g2', 'children'),
     Output('stats-card-g3', 'children'),
     Output('stats-card-g4', 'children'),
     Output('stats-card-g5', 'children')],
    [Input('analysis-goal', 'value'),
     Input('viz-num-feature', 'value'), Input('viz-plot-type-g1', 'value'),
     Input('viz-cat-feature', 'value'), Input('viz-plot-type-g2', 'value'),
     Input('viz-x-g3', 'value'), Input('viz-y-g3', 'value'), Input('viz-color-g3', 'value'), Input('viz-plot-type-g3', 'value'),
     Input('viz-x-g4', 'value'), Input('viz-y-g4', 'value'), Input('viz-color-g4', 'value'), Input('viz-plot-type-g4', 'value'), Input('viz-barmode-g4', 'value'),
     Input('viz-plot-type-g5', 'value'),
     Input('stored-data', 'data')]
)
def update_main_viz(goal, num_feat, plot_g1, cat_feat, plot_g2, x_g3, y_g3, color_g3, plot_g3, x_g4, y_g4, color_g4, plot_g4, barmode_g4, plot_g5, data):
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.update_layout(title="Select Parameters")
    stats_g1, stats_g2, stats_g3, stats_g4, stats_g5 = "", "", "", "", ""

    if goal == 'group-1':
        if num_feat in df.columns:
            series = df[num_feat].dropna()
            stats_g1 = [
                html.H6(f"Stats: {num_feat}"),
                html.P(f"Mean: {series.mean():.2f}"), html.P(f"Median: {series.median():.2f}"),
                html.P(f"Std: {series.std():.2f}"), html.P(f"Skew: {series.skew():.2f}"), html.P(f"Kurtosis: {series.kurt():.2f}")
            ]
            try:
                if plot_g1 == 'dist':
                    fig = px.histogram(df, x=num_feat, marginal="rug", opacity=0.7, title=f"Distribution: {num_feat}", color_discrete_sequence=['#636EFA'])
                elif plot_g1 == 'kde':
                    kde = stats.gaussian_kde(series)
                    x_range = np.linspace(series.min(), series.max(), 500)
                    y_kde = kde(x_range)
                    fig = go.Figure(go.Scatter(x=x_range, y=y_kde, fill='tozeroy', mode='lines', line=dict(width=3, color='purple')))
                    fig.update_layout(title=f"KDE Plot (Filled): {num_feat}", yaxis_title="Density")
                elif plot_g1 == 'rug':
                    fig = px.strip(df, x=num_feat, title=f"Rug Plot: {num_feat}", color_discrete_sequence=['black'])
                    fig.update_traces(opacity=0.5)
                elif plot_g1 == 'qq':
                    (osm, osr), _ = stats.probplot(series, dist="norm")
                    fig = px.scatter(x=osm, y=osr, title=f"Q-Q: {num_feat}")
                    
                    slope, intercept, r, p, stderr = stats.linregress(osm, osr)
                    line_x = np.array([min(osm), max(osm)])
                    line_y = slope * line_x + intercept
                    
                    fig.add_shape(type="line", x0=line_x[0], y0=line_y[0], x1=line_x[1], y1=line_y[1], line=dict(color="red"))
            except Exception as e: fig = go.Figure().update_layout(title=f"Error: {str(e)}")

    elif goal == 'group-2':
        if cat_feat in df.columns:
            counts = df[cat_feat].value_counts()
            stats_g2 = [
                html.H6(f"Stats: {cat_feat}"), html.P(f"Count: {len(df)}"),
                html.P(f"Unique: {df[cat_feat].nunique()}"), html.P(f"Mode: {counts.idxmax()} ({counts.max()})"),
                html.Hr(), html.Pre(counts.head(5).to_string(), style={'fontSize': '10px'})
            ]
            try:
                if plot_g2 == 'count': fig = px.histogram(df, x=cat_feat, color=cat_feat, text_auto=True, title=f"Count Plot: {cat_feat}")
                elif plot_g2 == 'pie': fig = px.pie(df, names=cat_feat, title=f"Pie Chart: {cat_feat}")
            except Exception as e: fig = go.Figure().update_layout(title=f"Error: {str(e)}")

    elif goal == 'group-3':
        if x_g3 in df.columns and y_g3 in df.columns:
            # Stats G3
            try:
                # Calculate correlation only on clean numeric data
                df_clean = df[[x_g3, y_g3]].dropna()
                if len(df_clean) > 1:
                    corr = df_clean[x_g3].corr(df_clean[y_g3])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean[x_g3], df_clean[y_g3])
                    r_sq = r_value ** 2
                    stats_g3 = [
                        html.H6(f"Correlation Analysis"),
                        html.P(f"Pearson Corr: {corr:.4f}"),
                        html.P(f"R-Squared: {r_sq:.4f}"),
                        html.P(f"Slope: {slope:.4f}"),
                        html.P(f"P-Value: {p_value:.4e}")
                    ]
                else:
                    stats_g3 = "Not enough data for correlation."
            except: stats_g3 = "Error calc stats."

            # Plots G3
            try:
                if plot_g3 == 'reg':
                    # Manual Regression to avoid statsmodels dependency
                    fig = px.scatter(df, x=x_g3, y=y_g3, color=color_g3, title=f"Regression: {x_g3} vs {y_g3}", opacity=0.6)
                    df_clean = df[[x_g3, y_g3]].dropna()
                    if len(df_clean) > 1:
                        slope, intercept, _, _, _ = stats.linregress(df_clean[x_g3], df_clean[y_g3])
                        line_x = np.array([df_clean[x_g3].min(), df_clean[x_g3].max()])
                        line_y = slope * line_x + intercept
                        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='OLS Trendline', line=dict(color='red', width=3)))
                        
                elif plot_g3 == 'joint':
                    fig = px.scatter(df, x=x_g3, y=y_g3, color=color_g3, marginal_x="histogram", marginal_y="histogram", title=f"Joint Plot: {x_g3} vs {y_g3}", opacity=0.6)
                elif plot_g3 == 'hex':
                    fig = px.density_heatmap(df, x=x_g3, y=y_g3, title=f"Hexbin/Density: {x_g3} vs {y_g3}")
                elif plot_g3 == 'contour':
                    fig = px.density_contour(df, x=x_g3, y=y_g3, color=color_g3, title=f"Contour Plot: {x_g3} vs {y_g3}")
                elif plot_g3 == '3d':
                    # Use log_trip_duration as default Z if not specified, or just trip_duration
                    z_val = 'log_trip_duration' if 'log_trip_duration' in df.columns else y_g3
                    fig = px.scatter_3d(df, x=x_g3, y=y_g3, z=z_val, color=color_g3, opacity=0.7, title=f"3D Plot (Z={z_val})")
                elif plot_g3 == 'line':
                    # Aggregating for line plot to avoid noise
                    df_agg = df.groupby(x_g3)[y_g3].mean().reset_index()
                    fig = px.line(df_agg, x=x_g3, y=y_g3, title=f"Line Plot (Mean {y_g3})")
                elif plot_g3 == 'area':
                    df_agg = df.groupby(x_g3)[y_g3].mean().reset_index()
                    fig = px.area(df_agg, x=x_g3, y=y_g3, title=f"Area Plot (Mean {y_g3})")
            except Exception as e: fig = go.Figure().update_layout(title=f"Error: {str(e)}")

    elif goal == 'group-4':
        if x_g4 in df.columns and y_g4 in df.columns:
            # Stats G4 (Mean, Min, Max per Group)
            try:
                stats_df = df.groupby(x_g4)[y_g4].agg(['mean', 'max', 'min', 'count']).reset_index()
                stats_df = stats_df.sort_values('mean', ascending=False)
                stats_g4 = [
                    html.H6(f"Stats: {y_g4} by {x_g4}"),
                    html.Div([
                        dash_table.DataTable(
                            data=stats_df.round(2).to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in stats_df.columns],
                            style_table={'overflowX': 'auto'},
                            page_size=5,
                            style_cell={'fontSize': '10px', 'textAlign': 'left'}
                        )
                    ])
                ]
            except: stats_g4 = "Error calculating stats."

            # Plots G4
            try:
                if plot_g4 == 'bar':
                    # Pre-aggregate for Bar Plot of Means
                    if color_g4: df_agg = df.groupby([x_g4, color_g4])[y_g4].mean().reset_index()
                    else: df_agg = df.groupby(x_g4)[y_g4].mean().reset_index()
                    
                    fig = px.bar(df_agg, x=x_g4, y=y_g4, color=color_g4, barmode=barmode_g4,
                                 title=f"Bar Plot (Mean {y_g4}): {x_g4} vs {y_g4}")
                                 
                elif plot_g4 == 'box':
                    fig = px.box(df, x=x_g4, y=y_g4, color=color_g4, title=f"Box Plot: {y_g4} by {x_g4}")
                    
                elif plot_g4 == 'violin':
                    fig = px.violin(df, x=x_g4, y=y_g4, color=color_g4, box=True, points='all',
                                    title=f"Violin Plot: {y_g4} by {x_g4}")
                                    
                elif plot_g4 == 'boxen':
                    fig = px.violin(df, x=x_g4, y=y_g4, color=color_g4, box=True, points=False,
                                    title=f"Multivariate Boxen (Violin-Style): {y_g4} by {x_g4}")
                                    
                elif plot_g4 == 'strip':
                    fig = px.strip(df, x=x_g4, y=y_g4, color=color_g4, title=f"Strip Plot: {y_g4} by {x_g4}")
                    
                elif plot_g4 == 'swarm':
                    # Mapping to Strip with overlay/jitter for Swarm effect
                    fig = px.strip(df, x=x_g4, y=y_g4, color=color_g4, stripmode='overlay', 
                                   title=f"Swarm/Strip Plot: {y_g4} by {x_g4}")
                    fig.update_traces(jitter=0.7)
            except Exception as e: fig = go.Figure().update_layout(title=f"Error: {str(e)}")

    elif goal == 'group-5':
        # Stats G5: Summary of highest correlation
        try:
            # Compute correlation matrix for all numerical columns
            corr_mat = df[numerical_cols].corr()
            
            # Find highest absolute correlation (excluding diagonal 1.0)
            np.fill_diagonal(corr_mat.values, np.nan)
            
            # Unstack to find max
            pairs = corr_mat.abs().unstack()
            pairs = pairs.sort_values(ascending=False)
            
            # Top pair
            top_pair = pairs.index[0]
            top_corr = corr_mat.loc[top_pair[0], top_pair[1]]
            
            stats_g5 = [
                html.H6("Global Statistics"),
                html.P(f"Strongest Correlation found between:"),
                html.B(f"{top_pair[0]} & {top_pair[1]}"),
                html.P(f"Correlation Coefficient: {top_corr:.4f}"),
                html.P("Note: 1.0 is perfect correlation.")
            ]
        except: stats_g5 = "Error calculating matrix stats."
        
        # Plots G5
        try:
            if plot_g5 == 'heatmap':
                fig = px.imshow(df[numerical_cols].corr(), text_auto=True, aspect="auto", 
                              color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            elif plot_g5 == 'pair':
                # Scatter Matrix (Limit to first 4 numerical cols to avoid browser lag)
                fig = px.scatter_matrix(df, dimensions=numerical_cols[:5], 
                                      title="Pair Plot (Top 5 Num Features)")
                fig.update_traces(diagonal_visible=False)
        except Exception as e: fig = go.Figure().update_layout(title=f"Error: {str(e)}")

    return fig, stats_g1, stats_g2, stats_g3, stats_g4, stats_g5

# Hypothesis Test Callback
@app.callback(
    Output('hyp-test-results', 'children'),
    [Input('stored-data', 'data'),
     Input('hyp-col', 'value'),
     Input('hyp-vendor', 'value'),
     Input('hyp-test-method', 'value')]
)
def render_hyp_test(data_json, feature, vendor, method):
    df = pd.DataFrame(data_json)
    
    if vendor != 'all':
        series = df[df['vendor_id'] == vendor][feature].dropna()
    else:
        series = df[feature].dropna()

    if len(series) < 3:
        return html.P("Not enough data to perform test.")

    stat_val, p_val = 0.0, 0.0

    try:
        if method == 'ks':
            z_scores = (series - series.mean()) / series.std()
            stat_val, p_val = stats.kstest(z_scores, 'norm')
        elif method == 'shapiro':
            stat_val, p_val = stats.shapiro(series)
        elif method == 'k2':
            stat_val, p_val = stats.normaltest(series)
    except Exception as e:
        return html.P(f"Error in calculation: {str(e)}")

    alpha_threshold = 0.01
    is_normal = p_val > alpha_threshold
    status_text = "PASS (Normal Distribution)" if is_normal else "FAIL (Not Normal)"
    alert_color = "success" if is_normal else "danger"

    return html.Div([
        html.H4(f"Test Outcome: {method.upper()}", className="fw-bold"),
        html.Hr(),
        html.H5(f"Feature: {feature}"),
        html.H6(f"Test Statistic: {stat_val:.4f}"),
        html.H6(f"P-Value: {p_val:.4f}"),
        html.P(f"Significance Level (α): {alpha_threshold}"),
        html.Div(f"Result: {status_text}", style={'padding': '10px', 'color': 'white', 'backgroundColor': '#28a745' if is_normal else '#dc3545', 'borderRadius': '5px', 'fontWeight': 'bold', 'textAlign': 'center'})
    ])


if __name__ == '__main__':
    app.run(debug=True)