# dash_app.py
import dash
from dash import dcc, html
from pathlib import Path
import plotly.io as pio
import json

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "data" / "processed" / "eda"

json_files = sorted(EDA_DIR.glob("*.json"))
plots = []

for json_file in json_files:
    try:
        fig_dict = json.loads(json_file.read_text())

        # Ignore summary/stat JSON files that are not Plotly figures
        if not isinstance(fig_dict, dict) or "data" not in fig_dict or "layout" not in fig_dict:
            print(f"Skipping {json_file.name}: not a Plotly figure JSON")
            continue

        # Clean invalid trace props if needed
        for trace in fig_dict.get("data", []):
            trace.pop("n", None)
            trace.pop("xaxis", None)
            trace.pop("yaxis", None)

        fig = pio.from_json(json.dumps(fig_dict))

        plots.append(
            (
                json_file,
                dcc.Graph(
                    figure=fig,
                    style={"height": "650px"}
                )
            )
        )

    except Exception as e:
        print(f"Skipping {json_file.name}: {e}")
        continue

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("EDA Dashboard", style={"textAlign": "center"}),
    html.Div(f"Loaded {len(plots)} plots successfully"),
    dcc.Tabs([
        dcc.Tab(
            label=json_file.stem.replace("_", " ").title(),
            children=[graph]
        )
        for json_file, graph in plots
    ])
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
