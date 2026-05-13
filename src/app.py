# app.py

import dash
from dash import dcc, html
from pathlib import Path
import plotly.io as pio
import json
from .styles import *
from dash.dependencies import Input, Output

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "data" / "processed" / "eda"

# ==========================================================
# CONFIGURAÇÃO DOS GRÁFICOS
# ==========================================================

GRAPH_METADATA = {
    "scatter_gasto_atividade_raw": {
        "title": "Relação entre Gastos e Atividade Parlamentar",
        "description": (
            "Deputados acima da linha de tendência podem indicar "
            "alto custo proporcional à atividade parlamentar."
        ),
        "section": "overview"
    },

    "heatmap_correlacoes": {
        "title": "Correlações entre Indicadores Parlamentares",
        "description": (
            "Mostra relações entre gastos, atividade, despesas "
            "e indicadores de desempenho."
        ),
        "section": "overview"
    },

    "bar_top_partidos_gasto_total": {
        "title": "Partidos com Maior Volume de Gastos",
        "description": (
            "Comparação entre os partidos com maior gasto agregado."
        ),
        "section": "partidos"
    },

    "boxplot_gasto_por_partido_top10": {
        "title": "Distribuição de Gastos por Partido",
        "description": (
            "Permite identificar dispersão, consistência e outliers."
        ),
        "section": "partidos"
    },
<<<<<<< HEAD
    "🎯 Análise Multidimensional": { #changing the name was the recommendation of the teacher
        "title": "Análises Multivariadas e Hierárquicas",
        "keywords": [
            "parallel_coordinates",
            "dendrograma_partidos",
            "heatmap_clusterizado",
            "treemap_partido_categoria",
            "treemap_partido_uf_deputado"
        ]
=======

    "bar_top_ufs_gasto_medio": {
        "title": "UFs com Maior Gasto Médio",
        "description": (
            "Comparação do gasto médio entre estados."
        ),
        "section": "ufs"
>>>>>>> 48f1f81 (novo design)
    },

    "stacked_area_partido_tempo": {
        "title": "Evolução Temporal dos Gastos",
        "description": (
            "Evolução dos gastos parlamentares ao longo do tempo."
        ),
        "section": "temporal"
    },

    "mapa_choropleth_gastos_uf": {
        "title": "Distribuição Geográfica dos Gastos",
        "description": (
            "Mapa dos gastos parlamentares agregados por estado."
        ),
        "section": "temporal"
    },

    "heatmap_partido_categoria_log": {
        "title": "Composição de Gastos por Categoria",
        "description": (
            "Identifica quais categorias concentram despesas "
            "em cada partido."
        ),
        "section": "advanced"
    },

    "treemap_partido_categoria": {
        "title": "Hierarquia de Gastos Parlamentares",
        "description": (
            "Visualização hierárquica das despesas "
            "por partido, categoria e deputado."
        ),
        "section": "advanced"
    },

    "parallel_coordinates_perfis": {
        "title": "Perfis Multivariados dos Partidos",
        "description": (
            "Comparação simultânea entre múltiplas métricas."
        ),
        "section": "advanced"
    },

    "heatmap_clusterizado_partidos": {
        "title": "Clusterização de Perfis Parlamentares",
        "description": (
            "Agrupa partidos com comportamento semelhante."
        ),
        "section": "advanced"
    },
}

# ==========================================================
# LOAD FIGURES
# ==========================================================

json_files = sorted(EDA_DIR.glob("*.json"))
figures = {}

for json_file in json_files:
    try:
        fig_dict = json.loads(json_file.read_text())

        if "data" not in fig_dict:
            continue

        fig = pio.from_json(json.dumps(fig_dict))

        figures[json_file.stem] = fig

    except Exception:
        continue

# ==========================================================
# APP
# ==========================================================

app = dash.Dash(__name__)

# ==========================================================
# HELPERS
# ==========================================================

def create_graph_card(graph_key):

    if graph_key not in figures:
        return html.Div()

    meta = GRAPH_METADATA.get(graph_key, {})

    return html.Div([

        html.H3(
            meta.get("title", graph_key),
            style=GRAPH_TITLE
        ),

        html.P(
            meta.get("description", ""),
            style=GRAPH_DESCRIPTION
        ),

        dcc.Graph(
            figure=figures[graph_key],
            style={"height": "480px"},
            config={"displayModeBar": False}
        )

    ], style=GRAPH_CARD)


def build_section(title, description, graph_keys):

    return html.Div([

        html.H2(title, style=SECTION_TITLE),

        html.P(description, style=SECTION_DESCRIPTION),

        html.Div(
            [create_graph_card(g) for g in graph_keys],
            style=GRAPH_GRID
        )

    ])


# ==========================================================
# KPIs
# ==========================================================

kpis = [
    ("Gasto Total", "R$ 1,2 bi"),
    ("Média por Deputado", "R$ 2,1 mi"),
    ("Maior Partido", "PL"),
    ("UF com Maior Média", "DF"),
]

kpi_cards = html.Div([

    html.Div([
        html.Div(label, style=KPI_LABEL),
        html.Div(value, style=KPI_VALUE),
    ], style=KPI_CARD)

    for label, value in kpis

], style=KPI_GRID)

# ==========================================================
# SIDEBAR
# ==========================================================

sidebar = html.Div([
    html.Div("🏛️ Radar Parlamentar", style=SIDEBAR_TITLE),
    html.Div("Plataforma de análise de gastos", style=SIDEBAR_SUBTITLE),

    html.Div([
        html.Div("NAVEGAÇÃO", style=NAV_TITLE),
        
        html.A("Visão Geral", href="#visao-geral", style=NAV_LINK),
        html.A("Análise por Partido", href="#analise-partido", style=NAV_LINK),
        html.A("Análise Regional", href="#analise-regional", style=NAV_LINK),
        html.A("Evolução Temporal", href="#evolucao-temporal", style=NAV_LINK),
        html.A("Investigação Avançada", href="#investigacao-avancada", style=NAV_LINK),
        
    ], style=NAV_SECTION),
], style=SIDEBAR_STYLE)

# ==========================================================
# LAYOUT
# ==========================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False), 
    sidebar,
    html.Div([
        html.Div(build_section(
            "Visão Geral",
            "Principais relações entre gastos parlamentares e atividade legislativa.",
            ["scatter_gasto_atividade_raw", "heatmap_correlacoes"]
        ), id="visao-geral"),

        html.Div(build_section(
            "Análise por Partido",
            "Comparação entre partidos políticos e padrões de despesa.",
            ["bar_top_partidos_gasto_total", "boxplot_gasto_por_partido_top10"]
        ), id="analise-partido"),

        html.Div(build_section(
            "Análise Regional",
            "Distribuição geográfica e comparação entre unidades federativas.",
            ["bar_top_ufs_gasto_medio", "mapa_choropleth_gastos_uf"]
        ), id="analise-regional"),

        html.Div(build_section(
            "Evolução Temporal",
            "Mudanças nos padrões de gastos ao longo do tempo.",
            ["stacked_area_partido_tempo"]
        ), id="evolucao-temporal"),

        html.Div(build_section(
            "Investigação Avançada",
            "Análises multivariadas, clustering e exploração de padrões complexos.",
            [
                "heatmap_partido_categoria_log",
                "treemap_partido_categoria",
                "parallel_coordinates_perfis",
                "heatmap_clusterizado_partidos",
            ]
        ), id="investigacao-avancada"),

    ], style=CONTENT_STYLE)
], style=APP_STYLE)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
