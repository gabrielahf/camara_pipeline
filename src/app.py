# dash_app.py
import dash
from dash import dcc, html
from pathlib import Path
import plotly.io as pio
import json

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "data" / "processed" / "eda"

# ==========================================================
# CONFIGURAÇÃO DAS ABAS (AGRUPAMENTO TEMÁTICO)
# ==========================================================
PLOT_GROUPS = {
    "📊 Visão Geral": {
        "title": "Distribuições e Métricas Gerais",
        "keywords": ["hist_", "boxplots_metricas", "heatmap_correlacoes"]
    },
    "🏛️ Por Partido": {
        "title": "Análise por Partido e UF",
        "keywords": [
            "bar_top_partidos",
            "boxplot_gasto_por_partido",
            "bar_top_ufs",
            "beeswarm_atividade"
        ]
    },
    "👤 Por Deputado": {
        "title": "Análise Individual",
        "keywords": ["scatter_gasto_atividade"]
    },
    "📈 Análise Temporal": {
        "title": "Evolução ao Longo do Tempo",
        "keywords": ["stacked_area"]
    },
    "🎯 Análise Avançada": {
        "title": "Análises Multivariadas e Hierárquicas",
        "keywords": [
            "parallel_coordinates",
            "dendrograma_partidos",
            "heatmap_clusterizado",
            "treemap_partido_categoria",
            "treemap_partido_uf_deputado"
        ]
    },
    "💰 Composição de Gastos": {
        "title": "Composição por Categoria de Despesa",
        "keywords": ["heatmap_partido_categoria"]
    }
}

def categorize_plot(filename: str) -> str:
    """Determina a qual grupo um plot pertence com base no nome do arquivo."""
    name_lower = filename.lower()
    for group_name, group_info in PLOT_GROUPS.items():
        if any(keyword in name_lower for keyword in group_info["keywords"]):
            return group_name
    return "📊 Visão Geral"

# ==========================================================
# CARREGAMENTO DOS GRÁFICOS
# ==========================================================
json_files = sorted(EDA_DIR.glob("*.json"))
plots_by_group = {group: [] for group in PLOT_GROUPS.keys()}
loaded_count = 0
skipped_count = 0

for json_file in json_files:
    try:
        fig_dict = json.loads(json_file.read_text())

        # Pular arquivos JSON que não são figuras do Plotly
        if not isinstance(fig_dict, dict) or "data" not in fig_dict or "layout" not in fig_dict:
            print(f"Skipping {json_file.name}: not a Plotly figure JSON")
            skipped_count += 1
            continue

        # Limpar propriedades inválidas
        for trace in fig_dict.get("data", []):
            trace.pop("n", None)
            trace.pop("xaxis", None)
            trace.pop("yaxis", None)

        fig = pio.from_json(json.dumps(fig_dict))
        group = categorize_plot(json_file.stem)
        
        plots_by_group[group].append(
            (
                json_file.stem,
                dcc.Graph(
                    figure=fig,
                    style={"height": "650px", "margin": "0"},
                    config={"displayModeBar": True}
                )
            )
        )
        loaded_count += 1

    except Exception as e:
        print(f"Skipping {json_file.name}: {e}")
        skipped_count += 1
        continue

# Remover grupos vazios
plots_by_group = {k: v for k, v in plots_by_group.items() if v}

# ==========================================================
# INTERFACE DO DASHBOARD
# ==========================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

def build_group_tab(group_name: str, plots: list):
    """Cria o conteúdo de uma aba com múltiplos gráficos."""
    group_info = PLOT_GROUPS.get(group_name, {"title": group_name})
    
    # ✅ CORREÇÃO: List comprehension corretamente encapsulada
    plot_cards = [
        html.Div([
            html.H3(
                plot_name.replace("_", " ").title(),
                style={
                    "color": "#34495e",
                    "margin": "0 0 15px 0",
                    "paddingLeft": "15px",
                    "borderLeft": "5px solid #3498db",
                    "fontSize": "1.2rem"
                }
            ),
            plot_graph
        ], style={
            "marginBottom": "30px",
            "padding": "20px",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"
        })
        for plot_name, plot_graph in plots
    ]

    return html.Div([
        html.H2(
            group_info["title"],
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "margin": "30px 0 20px 0",
                "paddingBottom": "10px",
                "borderBottom": "3px solid #3498db"
            }
        ),
        html.Div(plot_cards, style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"})
    ])

app.layout = html.Div([
    # Header
    html.Div([
        html.H1(
            "📊 Dashboard de Análise Exploratória (EDA)",
            style={
                "textAlign": "center",
                "padding": "30px 20px",
                "backgroundColor": "#3498db",
                "color": "white",
                "marginBottom": "0",
                "fontSize": "28px"
            }
        ),
        html.P(
            f"Carregados {loaded_count} gráficos com sucesso | {skipped_count} ignorados",
            style={
                "textAlign": "center",
                "padding": "10px",
                "backgroundColor": "#2980b9",
                "color": "#ecf0f1",
                "margin": "0",
                "fontWeight": "bold"
            }
        )
    ]),
    
    # Abas Temáticas
    dcc.Tabs(
        id="main-tabs",
        value=list(plots_by_group.keys())[0] if plots_by_group else None,
        children=[
            dcc.Tab(
                label=group_name,
                value=group_name,
                children=[build_group_tab(group_name, plots)],
                style={"padding": "15px 25px", "fontSize": "16px"},
                selected_style={"backgroundColor": "#e8f4f8", "borderTop": "4px solid #3498db"}
            )
            for group_name, plots in plots_by_group.items()
        ],
        style={"marginTop": "20px"}
    )
], style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#f4f6f9", "height": "100vh"})

if __name__ == "__main__":
    app.run(debug=True, port=8050)