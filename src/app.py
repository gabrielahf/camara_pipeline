# dash_app.py
import dash
from dash import dcc, html
from pathlib import Path
import plotly.io as pio
import json

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "data" / "processed" / "eda"

# Define thematic groups for plots
PLOT_GROUPS = {
    "📊 Visão Geral": {
        "title": "Distribuições e Métricas Gerais",
        "keywords": ["hist", "boxplot", "resumo", "correlacao", "heatmap_correlacoes"]
    },
    "🏛️ Por Partido": {
        "title": "Análise por Partido",
        "keywords": ["partido", "bar_top_partidos", "boxplot_gasto_por_partido", "beeswarm_atividade"]
    },
    "👤 Por Deputado": {
        "title": "Análise Individual por Deputado",
        "keywords": ["scatter_gasto_atividade", "treemap_partido_uf_deputado", "anomalias"]
    },
    "📈 Análise Temporal": {
        "title": "Evolução ao Longo do Tempo",
        "keywords": ["stacked_area", "tempo", "time"]
    },
    "🎯 Análise Avançada": {
        "title": "Análises Multivariadas e Clusterização",
        "keywords": ["parallel", "dendrogram", "clusterizado", "treemap_partido_categoria"]
    },
    "💰 Composição de Gastos": {
        "title": "Composição por Categoria de Despesa",
        "keywords": ["heatmap_partido_categoria", "categoria"]
    }
}

def categorize_plot(filename: str) -> str:
    """Determine which group a plot belongs to based on its filename."""
    name_lower = filename.lower()
    
    for group_name, group_info in PLOT_GROUPS.items():
        if any(keyword in name_lower for keyword in group_info["keywords"]):
            return group_name
    
    # Default group for uncategorized plots
    return "📊 Visão Geral"

# Load all plots
json_files = sorted(EDA_DIR.glob("*.json"))
plots_by_group = {group: [] for group in PLOT_GROUPS.keys()}
loaded_count = 0
skipped_count = 0

for json_file in json_files:
    try:
        fig_dict = json.loads(json_file.read_text())

        # Skip non-Plotly JSON files
        if not isinstance(fig_dict, dict) or "data" not in fig_dict or "layout" not in fig_dict:
            print(f"Skipping {json_file.name}: not a Plotly figure JSON")
            skipped_count += 1
            continue

        # Clean invalid trace props
        for trace in fig_dict.get("data", []):
            trace.pop("n", None)
            trace.pop("xaxis", None)
            trace.pop("yaxis", None)

        fig = pio.from_json(json.dumps(fig_dict))
        
        # Determine which group this plot belongs to
        group = categorize_plot(json_file.stem)
        
        plots_by_group[group].append(
            (
                json_file.stem,
                dcc.Graph(
                    figure=fig,
                    style={"height": "650px", "margin": "20px 0"},
                    config={"displayModeBar": True}
                )
            )
        )
        loaded_count += 1

    except Exception as e:
        print(f"Skipping {json_file.name}: {e}")
        skipped_count += 1
        continue

# Remove empty groups
plots_by_group = {k: v for k, v in plots_by_group.items() if v}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Build tab content for each group
def build_group_tab(group_name: str, plots: list):
    """Create a tab with multiple plots grouped together."""
    group_info = PLOT_GROUPS.get(group_name, {"title": group_name})
    
    return html.Div([
        html.H2(
            group_info["title"],
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "margin": "30px 0 20px 0",
                "padding": "10px",
                "borderBottom": "3px solid #3498db"
            }
        ),
        html.P(
            f"{len(plots)} visualizações nesta categoria",
            style={"textAlign": "center", "color": "#7f8c8d", "marginBottom": "30px"}
        ),
        html.Div([
            html.Div([
                html.H3(
                    plot_name.replace("_", " ").title(),
                    style={
                        "color": "#34495e",
                        "margin": "30px 0 15px 0",
                        "padding": "10px",
                        "backgroundColor": "#ecf0f1",
                        "borderLeft": "5px solid #3498db"
                    }
                ),
                plot_graph
            ], style={"marginBottom": "40px"})
            for plot_name, plot_graph in plots
        ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 20px"})
    ])

# Create main tabs
app.layout = html.Div([
    # Header
    html.Div([
        html.H1(
            "📊 Dashboard de Análise Exploratória",
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "padding": "30px",
                "backgroundColor": "#3498db",
                "color": "white",
                "marginBottom": "0"
            }
        ),
        html.P(
            f"Carregados {loaded_count} gráficos com sucesso | {skipped_count} arquivos ignorados",
            style={
                "textAlign": "center",
                "padding": "15px",
                "backgroundColor": "#2980b9",
                "color": "white",
                "margin": "0",
                "fontSize": "16px"
            }
        )
    ]),
    
    # Navigation Tabs
    dcc.Tabs(
        id="main-tabs",
        value=list(plots_by_group.keys())[0] if plots_by_group else None,
        children=[
            dcc.Tab(
                label=group_name,
                value=group_name,
                children=[build_group_tab(group_name, plots)],
                style={"padding": "15px 25px", "fontSize": "16px", "fontWeight": "bold"},
                selected_style={"backgroundColor": "#ecf0f1", "borderTop": "4px solid #3498db"}
            )
            for group_name, plots in plots_by_group.items()
        ],
        style={"marginBottom": "30px"}
    )
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f8f9fa"})

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")