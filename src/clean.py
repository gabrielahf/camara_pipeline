from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(exist_ok=True)


def clean_deputados() -> pd.DataFrame:
    """
    Lê e limpa a base de deputados.
    - Remove duplicados
    - Mantém apenas colunas principais
    """
    df = pd.read_parquet(DATA_RAW / "deputados.parquet")
    df = df.drop_duplicates(subset="id")

    cols = ["idDeputado", "nome", "siglaPartido", "siglaUf"]
    df = df[cols]

    df.to_parquet(DATA_PROC / "deputados_clean.parquet", index=False)
    return df


def clean_despesas() -> pd.DataFrame:
    """
    Lê e limpa a base de despesas.
    - Garante tipo numérico em valorLiquido
    - Remove registros sem valor
    - Cria colunas auxiliares para lidar com negativos (estornos)
    """
    df = pd.read_parquet(DATA_RAW / "despesas.parquet")

    # Remove duplicados “brutos”
    df = df.drop_duplicates()

    # Converte valorLiquido para float
    df["valorLiquido"] = pd.to_numeric(df["valorLiquido"], errors="coerce")

    # Remove linhas sem valor
    df = df[~df["valorLiquido"].isna()]

    # Cria coluna com gasto só positivo (para análises de “quanto foi gasto”)
    df["valorLiquido_pos"] = df["valorLiquido"].clip(lower=0)

    # Marca se é estorno / ajuste
    df["eh_estorno"] = df["valorLiquido"] < 0

    # Garante idDeputado numérico
    if "idDeputado" in df.columns:
        df["idDeputado"] = pd.to_numeric(df["idDeputado"], errors="coerce")

    df.to_parquet(DATA_PROC / "despesas_clean.parquet", index=False)
    return df


def clean_proposicoes() -> pd.DataFrame:
    """
    Lê e limpa a base de proposições.
    - Remove duplicados por idProposicao + idDeputado (se existir)
    - Garante tipo numérico do idDeputado
    """
    df = pd.read_parquet(DATA_RAW / "proposicoes.parquet")

    # Tenta encontrar a coluna id da proposição
    id_cols = [c for c in df.columns if c.lower().startswith("id")]
    if "idDeputado" in df.columns:
        key_cols = ["idDeputado"]
        if "id" in df.columns:
            key_cols.append("id")
        elif "idProposicao" in df.columns:
            key_cols.append("idProposicao")
        df = df.drop_duplicates(subset=key_cols)

        df["idDeputado"] = pd.to_numeric(df["idDeputado"], errors="coerce")
    else:
        # Se não tiver idDeputado, só remove duplicados gerais
        df = df.drop_duplicates()

    df.to_parquet(DATA_PROC / "proposicoes_clean.parquet", index=False)
    return df


def clean_eventos() -> pd.DataFrame:
    """
    Lê e limpa a base de eventos.
    - Remove duplicados por idEvento + idDeputado (se existir)
    - Garante tipo numérico do idDeputado
    """
    df = pd.read_parquet(DATA_RAW / "eventos.parquet")

    if "idDeputado" in df.columns:
        key_cols = ["idDeputado"]
        if "id" in df.columns:
            key_cols.append("id")
        elif "idEvento" in df.columns:
            key_cols.append("idEvento")
        df = df.drop_duplicates(subset=key_cols)

        df["idDeputado"] = pd.to_numeric(df["idDeputado"], errors="coerce")
    else:
        df = df.drop_duplicates()

    df.to_parquet(DATA_PROC / "eventos_clean.parquet", index=False)
    return df


def build_dataset_modelagem(
    df_deps: pd.DataFrame,
    df_desp: pd.DataFrame,
    df_prop: pd.DataFrame,
    df_evt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera o dataset final já limpo e pronto para modelagem.
    - Agrega gastos por deputado
    - Conta proposições e eventos
    - Cria métrica de atividade composta
    """

    # Base: deputados
    df = df_deps.copy()

    # Agregação de despesas (usando valorLiquido_pos para ignorar estornos)
    if not df_desp.empty and "idDeputado" in df_desp.columns:
        agg_desp = df_desp.groupby("idDeputado").agg(
            gasto_total=("valorLiquido_pos", "sum"),
            qtd_despesas=("valorLiquido_pos", "count"),
            gasto_liquido=("valorLiquido", "sum"),  # incluindo negativos
            qtd_estornos=("eh_estorno", "sum"),
        ).reset_index()
        df = df.merge(agg_desp, on="idDeputado", how="left")

    # Contagem de proposições
    if not df_prop.empty and "idDeputado" in df_prop.columns:
        agg_prop = df_prop.groupby("idDeputado").size().reset_index(
            name="total_proposicoes"
        )
        df = df.merge(agg_prop, on="idDeputado", how="left")

    # Contagem de eventos
    if not df_evt.empty and "idDeputado" in df_evt.columns:
        agg_evt = df_evt.groupby("idDeputado").size().reset_index(
            name="total_eventos"
        )
        df = df.merge(agg_evt, on="idDeputado", how="left")

    # Preenche NaN com 0 onde faz sentido
    for col in [
        "gasto_total",
        "qtd_despesas",
        "gasto_liquido",
        "qtd_estornos",
        "total_proposicoes",
        "total_eventos",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Cria métrica de atividade composta (pode ajustar os pesos depois)
    df["atividade_composta"] = df["total_proposicoes"] + df["total_eventos"] * 0.3

    # Salva dataset final de modelagem
    df.to_csv(DATA_PROC / "dataset_modelagem.csv", index=False)
    df.to_parquet(DATA_PROC / "dataset_modelagem.parquet", index=False)

    return df


def main():
    print("Limpando dados brutos...")

    df_deps = clean_deputados()
    print(f"Deputados limpos: {len(df_deps)}")

    df_desp = clean_despesas()
    print(f"Despesas limpas: {len(df_desp)}")

    df_prop = clean_proposicoes()
    print(f"Proposições limpas: {len(df_prop)}")

    df_evt = clean_eventos()
    print(f"Eventos limpos: {len(df_evt)}")

    print("\n Construindo dataset de modelagem...")
    df_model = build_dataset_modelagem(df_deps, df_desp, df_prop, df_evt)
    print(f"Dataset de modelagem gerado com {len(df_model)} deputados.")
    print("Arquivos salvos em data/processed/")


if __name__ == "__main__":
    main()
