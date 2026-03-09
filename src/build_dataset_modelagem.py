from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==== CONFIGURAÇÃO DE CAMINHOS ====

ROOT = Path(__file__).resolve().parents[1]  # pasta raiz do projeto
DATA_PROCESSED = ROOT / "data" / "processed"

DESPESAS_PATH = DATA_PROCESSED / "despesas_clean.parquet"
EVENTOS_PATH = DATA_PROCESSED / "eventos_clean.parquet"
PROPS_PATH = DATA_PROCESSED / "proposicoes_clean.parquet"

OUT_CSV = DATA_PROCESSED / "dataset_mestre_clean.csv"
OUT_PARQUET = DATA_PROCESSED / "dataset_mestre_clean.parquet"

DESP_CAT_CSV = DATA_PROCESSED / "despesas_categorizadas.csv"
PLOT_CAT = DATA_PROCESSED / "gastos_por_categoria.png"


# ==== CLASSIFICAÇÃO DE DESPESAS EM MACRO-CATEGORIAS ====


GRUPOS_CATEGORIA = {
    "Transporte": [
        "bilhete", "passagem", "locomoção", "locomocao", "transporte",
        "aluguel de veículos", "aluguel de veiculos",
        "combustível", "combustivel", "taxi", "uber", "veículo", "veiculo",
        "fretamento"
    ],
    "Alimentação": ["alimentação", "alimentacao", "restaurante", "refeição", "refeicao", "lanches"],
    "Escritório e Funcionamento": [
        "escritório", "escritorio", "materiais", "serviços postais",
        "correios", "telefone", "internet", "locação", "locacao",
        "espaço", "espaco", "condomínio", "condominio",
        "copiadora", "material de expediente"
    ],
    "Cursos e Capacitação": ["curso", "capacitação", "capacitação", "treinamento", "inscrição", "inscricao"],
    "Divulgação do Mandato": [
        "publicidade", "divulgação", "divulgacao",
        "assessoria de imprensa", "marketing"
    ],
}


def _classificar_macro_categoria(desc: str) -> str:
    if not isinstance(desc, str):
        return "Outros"
    desc = desc.lower()
    for categoria, palavras in GRUPOS_CATEGORIA.items():
        if any(p in desc for p in palavras):
            return categoria
    return "Outros"


def adicionar_categoria_macro(df_desp: pd.DataFrame) -> pd.DataFrame:
    """
    Garante a coluna 'categoria_macro' em df_desp, usando 'tipoDespesa'.
    Também salva um CSV detalhado com as despesas categorizadas.
    """
    print("\nClassificando despesas em macro-categorias...")

    if "tipoDespesa" not in df_desp.columns:
        print("  Aviso: coluna 'tipoDespesa' não encontrada em despesas_clean. "
              "Não será criada categoria_macro.")
        df_desp["categoria_macro"] = "Outros"
        return df_desp

    df = df_desp.copy()
    df["tipoDespesa"] = df["tipoDespesa"].astype(str).str.lower()
    df["categoria_macro"] = df["tipoDespesa"].apply(_classificar_macro_categoria)

    # salva CSV categorizado (para análises extras / dashboard)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DESP_CAT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ Arquivo de despesas categorizadas salvo em: {DESP_CAT_CSV}")

    return df


def plot_gastos_por_categoria(df_desp: pd.DataFrame) -> None:
    """
    Gera gráfico de barras de gastos por macro-categoria
    e salva em PNG.
    """
    print("\nGerando gráfico de gastos por categoria_macro...")

    if "categoria_macro" not in df_desp.columns:
        print("  Aviso: 'categoria_macro' não existe em df_desp. Gráfico não será gerado.")
        return

    if "valorLiquido" not in df_desp.columns:
        print("  Aviso: 'valorLiquido' não existe em df_desp. Gráfico não será gerado.")
        return

    df = df_desp.copy()
    df["valorLiquido"] = pd.to_numeric(df["valorLiquido"], errors="coerce").fillna(0.0)

    agrupado = (
        df.groupby("categoria_macro")["valorLiquido"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(agrupado.index, agrupado.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total gasto (R$)")
    plt.title("Gastos por Macro-Categoria (Câmara dos Deputados)")
    plt.tight_layout()

    plt.savefig(PLOT_CAT, dpi=150)
    plt.close()

    print(f"  ✅ Gráfico salvo em: {PLOT_CAT}")
    print(agrupado)


# ==== CARREGAMENTO ====

def load_parquets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Lendo arquivos limpos (.parquet)...")

    df_desp = pd.read_parquet(DESPESAS_PATH)
    df_evt = pd.read_parquet(EVENTOS_PATH)
    df_prop = pd.read_parquet(PROPS_PATH)

    print(f"  despesas_clean:    {len(df_desp):,} linhas")
    print(f"  eventos_clean:     {len(df_evt):,} linhas")
    print(f"  proposicoes_clean: {len(df_prop):,} linhas")

    return df_desp, df_evt, df_prop


# ==== AGREGAÇÕES ====

def agregar_despesas(df_desp: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega despesas por deputado (e por categoria_macro, se existir).
    Espera ao menos:
      - idDeputado
      - valorLiquido
      - (opcional) categoria_macro
    """
    print("\nAgregando despesas...")

    if "idDeputado" not in df_desp.columns:
        raise ValueError("Coluna 'idDeputado' não encontrada em despesas_clean.parquet")

    if "valorLiquido" not in df_desp.columns:
        raise ValueError("Coluna 'valorLiquido' não encontrada em despesas_clean.parquet")

    # garantir numérico
    df_desp["valorLiquido"] = pd.to_numeric(df_desp["valorLiquido"], errors="coerce").fillna(0.0)

    # diagnóstico de valores negativos
    n_negativos = (df_desp["valorLiquido"] < 0).sum()
    if n_negativos > 0:
        print(f"  Aviso: {n_negativos} registros de despesa com valor negativo (possíveis estornos)")

    # agregação básica por deputado
    agg = (
        df_desp
        .groupby("idDeputado", as_index=False)
        .agg(
            gasto_total=("valorLiquido", "sum"),
            qtd_despesas=("valorLiquido", "size"),
        )
    )

    # se tiver coluna de categoria macro, gera colunas wide por categoria
    if "categoria_macro" in df_desp.columns:
        print("  Encontrada coluna 'categoria_macro' -> agregando por categoria...")

        pivot_cat = (
            df_desp
            .groupby(["idDeputado", "categoria_macro"])["valorLiquido"]
            .sum()
            .reset_index()
            .pivot(index="idDeputado", columns="categoria_macro", values="valorLiquido")
            .fillna(0.0)
        )

        pivot_cat.columns = [
            f"gasto_{str(c).lower().replace(' ', '_')}"
            for c in pivot_cat.columns
        ]
        pivot_cat = pivot_cat.reset_index()

        agg = agg.merge(pivot_cat, on="idDeputado", how="left")

    return agg


def agregar_eventos(df_evt: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega eventos por deputado.
    Espera ao menos:
      - idDeputado
    """
    print("\nAgregando eventos...")

    if "idDeputado" not in df_evt.columns:
        raise ValueError("Coluna 'idDeputado' não encontrada em eventos_clean.parquet")

    agg_evt = (
        df_evt
        .groupby("idDeputado", as_index=False)
        .size()
        .rename(columns={"size": "total_eventos"})
    )

    return agg_evt


def agregar_proposicoes(df_prop: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega proposições por deputado.
    Espera ao menos:
      - idDeputado (ou coluna equivalente)
    """
    print("\nAgregando proposições...")

    if "idDeputado" not in df_prop.columns:
        possiveis = [c for c in df_prop.columns if "iddep" in c.lower() or "autor" in c.lower()]
        if possiveis:
            print(f"  Atenção: renomeando coluna '{possiveis[0]}' para 'idDeputado'")
            df_prop = df_prop.rename(columns={possiveis[0]: "idDeputado"})
        else:
            raise ValueError(
                "Nenhuma coluna parecida com 'idDeputado' encontrada em proposicoes_clean.parquet"
            )

    agg_prop = (
        df_prop
        .groupby("idDeputado", as_index=False)
        .size()
        .rename(columns={"size": "total_proposicoes"})
    )

    return agg_prop


# ==== DATASET MESTRE ====

def montar_dataset_mestre(
    df_desp_agg: pd.DataFrame,
    df_evt_agg: pd.DataFrame,
    df_prop_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta todas as agregações num único dataset mestre por deputado.
    """
    print("\nMontando dataset mestre...")

    ids = (
        pd.concat(
            [
                df_desp_agg[["idDeputado"]],
                df_evt_agg[["idDeputado"]],
                df_prop_agg[["idDeputado"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
    )

    df = ids.copy()
    df = df.merge(df_desp_agg, on="idDeputado", how="left")
    df = df.merge(df_evt_agg, on="idDeputado", how="left")
    df = df.merge(df_prop_agg, on="idDeputado", how="left")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    for col in ["gasto_total", "qtd_despesas", "total_eventos", "total_proposicoes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "total_proposicoes" in df.columns and "total_eventos" in df.columns:
        df["atividade_composta"] = (
            df["total_proposicoes"].astype(float)
            + 0.3 * df["total_eventos"].astype(float)
        )
    else:
        print("  Aviso: não foi possível criar 'atividade_composta' (faltam colunas).")
        df["atividade_composta"] = 0.0

    # gasto_total_ajustado
    if "gasto_total" in df.columns:
        df["gasto_total_ajustado"] = df["gasto_total"].clip(lower=0.0)
        n_negativos = (df["gasto_total"] < 0).sum()
        if n_negativos > 0:
            print(f"  Info: {n_negativos} deputados com gasto_total negativo (ajustados para 0 em 'gasto_total_ajustado')")
    else:
        df["gasto_total_ajustado"] = 0.0

    # flag sem despesa
    df["sem_despesa"] = (
        (df["gasto_total"].fillna(0.0) == 0.0)
        & (df["qtd_despesas"].fillna(0.0) == 0.0)
    )
    print(f"  Deputados sem despesa registrada: {df['sem_despesa'].sum()}")

    # custo por atividade
    df["custo_por_atividade"] = df["gasto_total_ajustado"] / (df["atividade_composta"] + 1e-6)

    return df


# ==== MAIN ====

def main():
    df_desp, df_evt, df_prop = load_parquets()

    # 1) garantir macro-categorias + salvar CSV detalhado
    df_desp = adicionar_categoria_macro(df_desp)

    # 2) gerar gráfico geral de gastos por macro-categoria
    plot_gastos_por_categoria(df_desp)

    # 3) agregações para dataset mestre
    df_desp_agg = agregar_despesas(df_desp)
    df_evt_agg = agregar_eventos(df_evt)
    df_prop_agg = agregar_proposicoes(df_prop)

    df_mestre = montar_dataset_mestre(df_desp_agg, df_evt_agg, df_prop_agg)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_mestre.to_csv(OUT_CSV, index=False)
    df_mestre.to_parquet(OUT_PARQUET, index=False)

    print("\n✅ Dataset mestre (limpo) gerado com sucesso!")
    print(f"  CSV:     {OUT_CSV}")
    print(f"  PARQUET: {OUT_PARQUET}")
    print(f"  Linhas:  {len(df_mestre):,}")


if __name__ == "__main__":
    main()
