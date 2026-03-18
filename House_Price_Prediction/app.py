
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# ── Load & prep data ──────────────────────────────────────────
df = pd.read_csv("data.csv")
df['zipcode'] = df['statezip'].str.split(' ').str[1].astype(int)
df = df.drop(columns=['date', 'street', 'statezip', 'country'])
df['house_age'] = 2014 - df['yr_built']
df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
df = df.drop(columns=['yr_built', 'yr_renovated'])

# Correlation with price
corr = df.select_dtypes(include=np.number).corr()['price'].drop('price').sort_values()

# PCA variance
variance_ratios = [0.2728, 0.1354, 0.1133, 0.0877, 0.0829,
                   0.0721, 0.0565, 0.0457, 0.0336, 0.0321, 0.0272]
cumulative = np.cumsum(variance_ratios)

# ── Figures ───────────────────────────────────────────────────

# 1. Price distribution
fig_price = make_subplots(rows=1, cols=2,
                          subplot_titles=["Raw price", "Log price"])
fig_price.add_trace(go.Histogram(x=df['price'], nbinsx=50,
                    marker_color='steelblue', name="Raw"), row=1, col=1)
fig_price.add_trace(go.Histogram(x=np.log(df['price']), nbinsx=50,
                    marker_color='coral', name="Log"), row=1, col=2)
fig_price.update_layout(title="Price distribution", showlegend=False)

# 2. Correlation bar chart
fig_corr = go.Figure(go.Bar(
    x=corr.values,
    y=corr.index,
    orientation='h',
    marker_color=['coral' if v < 0 else 'steelblue' for v in corr.values]
))
fig_corr.update_layout(title="Feature correlation with price",
                        xaxis_title="Correlation coefficient",
                        yaxis_title="Feature")

# 3. Correlation heatmap
corr_matrix = df.select_dtypes(include=np.number).corr()
fig_heatmap = go.Figure(go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu',
    zmid=0,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    textfont={"size": 9}
))
fig_heatmap.update_layout(title="Correlation heatmap")

# 4. City boxplot
city_order = df.groupby('city')['price'].median().sort_values(ascending=False).index
fig_city = px.box(df, x='city', y='price', category_orders={'city': list(city_order)},
                  color='city', title="House price by city")
fig_city.update_layout(showlegend=False, xaxis_tickangle=-45)

# 5. PCA variance
fig_pca = make_subplots(rows=1, cols=2,
                         subplot_titles=["Variance per component",
                                         "Cumulative explained variance"])
fig_pca.add_trace(go.Bar(x=list(range(1, 12)),
                          y=[v*100 for v in variance_ratios],
                          marker_color='steelblue', name="Per component"),
                  row=1, col=1)
fig_pca.add_trace(go.Scatter(x=list(range(1, 12)), y=cumulative*100,
                              mode='lines+markers', marker_color='coral',
                              name="Cumulative"),
                  row=1, col=2)
fig_pca.add_hline(y=95, line_dash="dash", line_color="gray",
                   annotation_text="95% threshold", row=1, col=2)
fig_pca.update_layout(title="PCA — Explained variance", showlegend=False)

# 6. Train vs test metrics
fig_metrics = go.Figure()
metrics_names = ["MAE", "RMSE", "R²"]
train_vals = [121342, 157036, 0.4016]
test_vals  = [119131, 151552, 0.4282]
fig_metrics.add_trace(go.Bar(name='Train', x=metrics_names, y=train_vals,
                              marker_color='steelblue'))
fig_metrics.add_trace(go.Bar(name='Test',  x=metrics_names, y=test_vals,
                              marker_color='coral'))
fig_metrics.update_layout(title="Train vs test performance",
                           barmode='group', yaxis_title="Score")

# ── Layout ────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("🏠 House Price Prediction",
                style={'color': 'white', 'margin': '0', 'fontSize': '2rem'}),
        html.P("Linear Regression + PCA from scratch | Washington State, USA",
               style={'color': '#cce', 'margin': '6px 0 0'})
    ], style={'background': '#1a1a2e', 'padding': '30px 40px'}),

    # KPI cards
    html.Div([
        html.Div([
            html.H3("Total Houses", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'}),
            html.H2(f"{len(df):,}", style={'margin': '4px 0 0', 'color': '#1a1a2e'})
        ], style={'background': 'white', 'padding': '20px 30px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                  'flex': '1', 'textAlign': 'center'}),

        html.Div([
            html.H3("Avg Price", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'}),
            html.H2(f"${df['price'].mean():,.0f}", style={'margin': '4px 0 0', 'color': '#1a1a2e'})
        ], style={'background': 'white', 'padding': '20px 30px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                  'flex': '1', 'textAlign': 'center'}),

        html.Div([
            html.H3("R² Score", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'}),
            html.H2("0.4282", style={'margin': '4px 0 0', 'color': '#2ecc71'})
        ], style={'background': 'white', 'padding': '20px 30px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                  'flex': '1', 'textAlign': 'center'}),

        html.Div([
            html.H3("MAE", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'}),
            html.H2("$119,131", style={'margin': '4px 0 0', 'color': '#e74c3c'})
        ], style={'background': 'white', 'padding': '20px 30px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                  'flex': '1', 'textAlign': 'center'}),

        html.Div([
            html.H3("RMSE", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'}),
            html.H2("$151,552", style={'margin': '4px 0 0', 'color': '#e67e22'})
        ], style={'background': 'white', 'padding': '20px 30px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                  'flex': '1', 'textAlign': 'center'}),

    ], style={'display': 'flex', 'gap': '16px', 'padding': '30px 40px',
              'background': '#f4f6f9'}),

    # Section: EDA
    html.Div([
        html.H2("1. Exploratory Data Analysis",
                style={'color': '#1a1a2e', 'borderBottom': '3px solid #4a90d9',
                       'paddingBottom': '10px'}),
        dcc.Graph(figure=fig_price),
        dcc.Graph(figure=fig_corr),
        dcc.Graph(figure=fig_heatmap),
        dcc.Graph(figure=fig_city),
    ], style={'padding': '30px 40px', 'background': 'white', 'margin': '0 0 16px'}),

    # Section: PCA
    html.Div([
        html.H2("2. PCA — Dimensionality Reduction",
                style={'color': '#1a1a2e', 'borderBottom': '3px solid #9b59b6',
                       'paddingBottom': '10px'}),
        html.Div([
            html.Div("14 features → 11 components retaining 95% variance",
                     style={'background': '#eaf4fb', 'padding': '12px 20px',
                            'borderLeft': '4px solid #4a90d9', 'borderRadius': '4px',
                            'marginBottom': '16px'}),
            html.Div("sqft_living & sqft_above had 0.84 correlation — PCA removed this redundancy",
                     style={'background': '#fef9e7', 'padding': '12px 20px',
                            'borderLeft': '4px solid #f39c12', 'borderRadius': '4px',
                            'marginBottom': '16px'}),
        ]),
        dcc.Graph(figure=fig_pca),
    ], style={'padding': '30px 40px', 'background': 'white', 'margin': '0 0 16px'}),

    # Section: Model
    html.Div([
        html.H2("3. Model Performance",
                style={'color': '#1a1a2e', 'borderBottom': '3px solid #2ecc71',
                       'paddingBottom': '10px'}),
        html.Div("Gradient Descent — 1000 iterations, learning rate = 0.01",
                 style={'background': '#eafaf1', 'padding': '12px 20px',
                        'borderLeft': '4px solid #2ecc71', 'borderRadius': '4px',
                        'marginBottom': '16px'}),
        dcc.Graph(figure=fig_metrics),
    ], style={'padding': '30px 40px', 'background': 'white', 'margin': '0 0 16px'}),

    # Section: Key findings
    html.Div([
        html.H2("4. Key Findings",
                style={'color': '#1a1a2e', 'borderBottom': '3px solid #e74c3c',
                       'paddingBottom': '10px'}),
        html.Ul([
            html.Li("sqft_living is the strongest predictor of price (correlation: 0.57)"),
            html.Li("Location (city/zipcode) significantly affects price"),
            html.Li("No overfitting — Train R² (0.40) ≈ Test R² (0.43)"),
            html.Li("R² of 0.43 suggests non-linear patterns exist — future work: polynomial features or Random Forest"),
        ], style={'lineHeight': '2', 'fontSize': '1rem'})
    ], style={'padding': '30px 40px', 'background': 'white'}),

    # Footer
    html.Div([
        html.P("Built with Python · NumPy · Pandas · Plotly Dash",
               style={'color': '#aaa', 'textAlign': 'center', 'margin': '0'})
    ], style={'background': '#1a1a2e', 'padding': '20px'})

], style={'fontFamily': 'Arial, sans-serif', 'background': '#f4f6f9', 'minHeight': '100vh'})

if __name__ == '__main__':
    app.run(debug=False, port=8050)
