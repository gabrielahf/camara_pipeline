from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"

def load_dataset():
    parquet = DATA_PROCESSED / "dataset_mestre.parquet"
    csv = DATA_PROCESSED / "dataset_mestre.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Nenhum dataset encontrado em data/processed/")

    return df

def main():
    df = load_dataset()

    # garante colunas numéricas
    for col in ["gasto_total", "total_proposicoes", "total_eventos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # cria atividade_composta se necessário
    if "atividade_composta" not in df.columns:
        df["atividade_composta"] = (
            df.get("total_proposicoes", 0)
            + df.get("total_eventos", 0) * 0.3
        )

    x = df["gasto_total"].astype(float)
    y = df["atividade_composta"].astype(float)

    r = float(np.corrcoef(x, y)[0,1]) if len(df) >= 2 else float("nan")

    # Scatter gasto vs atividade
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel("Gasto total (R$)")
    plt.ylabel("Atividade composta")
    plt.title("Relação entre gasto e atividade")
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / "scatter_gasto_atividade.png", dpi=130)
    plt.close()

    # K-Means clustering
    features = df[["gasto_total", "atividade_composta"]].copy()
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    inertias = []
    models = []

    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        models.append(km)

    best_k = int(np.argmin(inertias)) + 2
    best_model = models[best_k - 2]
    df["cluster"] = best_model.labels_

    resumo = df.groupby("cluster").agg(
        n=("idDeputado", "count"),
        gasto_medio=("gasto_total", "mean"),
        atividade_media=("atividade_composta", "mean")
    ).reset_index().to_dict(orient="records")

    # Scatter clusters
    plt.figure()
    for c in sorted(df["cluster"].unique()):
        m = df["cluster"] == c
        plt.scatter(df.loc[m, "gasto_total"], df.loc[m, "atividade_composta"], alpha=0.6, label=f"Cluster {c}")
    plt.legend()
    plt.xlabel("Gasto total (R$)")
    plt.ylabel("Atividade composta")
    plt.title("Perfis de parlamentares (clusters KMeans)")
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / "clusters_scatter.png", dpi=130)
    plt.close()

    (DATA_PROCESSED / "correlacao_pearson.json").write_text(
        json.dumps({"pearson_r": r}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA_PROCESSED / "kmeans_resumo.json").write_text(
        json.dumps({"best_k": best_k, "resumo": resumo}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Correlação (r) =", r)
    print("Melhor k =", best_k)
    print("Arquivos salvos em", DATA_PROCESSED)

if __name__ == "__main__":
    main()
