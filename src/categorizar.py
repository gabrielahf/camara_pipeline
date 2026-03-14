import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Caminhos
ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "despesas.parquet"
PROC = ROOT / "data" / "processed" / "despesas_categorizadas.csv"
PLOT = ROOT / "data" / "processed" / "gastos_por_categoria.png"

# RAW = Path("data/raw/despesas.parquet")
# PROC = Path("data/processed/despesas_categorizadas.csv")
# PLOT = Path("data/processed/gastos_por_categoria.png")


# ---------------------------
# 1) Função de categorização
# ---------------------------


def categorizar_despesas():
    print("Carregando despesas...")
    df = pd.read_parquet(RAW)

    df["tipoDespesa"] = df["tipoDespesa"].str.lower()

    grupos = {
        "Transporte": [
            "bilhete",
            "passagem",
            "locomoção",
            "transporte",
            "aluguel de veículos",
            "combustível",
            "taxi",
            "uber",
            "veículo",
            "fretamento",
        ],
        "Alimentação": ["alimentação", "restaurante", "refeição", "lanches"],
        "Escritório e Funcionamento": [
            "escritório",
            "materiais",
            "serviços postais",
            "correios",
            "telefone",
            "internet",
            "locação",
            "espaço",
            "condomínio",
            "copiadora",
            "material de expediente",
        ],
        "Cursos e Capacitação": ["curso", "capacitação", "treinamento", "inscrição"],
        "Divulgação do Mandato": [
            "publicidade",
            "divulgação",
            "assessoria de imprensa",
            "marketing",
        ],
    }

    def classificar(desc):
        for categoria, palavras in grupos.items():
            if any(palavra in desc for palavra in palavras):
                return categoria
        return "Outros"

    df["macroCategoria"] = df["tipoDespesa"].apply(classificar)
    df.to_csv(PROC, index=False, encoding="utf-8-sig")

    print(f"✅ Arquivo gerado: {PROC}")
    return df


# ---------------------------
# 2) Função para gerar gráfico
# ---------------------------


def plot_gastos_por_categoria():
    print("Gerando gráfico de gastos por categoria...")

    df = pd.read_csv(PROC)

    agrupado = (
        df.groupby("macroCategoria")["valorLiquido"].sum().sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(agrupado.index, agrupado.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total gasto (R$)")
    plt.title("Gastos por Macro-Categoria (Câmara dos Deputados)")
    plt.tight_layout()

    plt.savefig(PLOT, dpi=150)
    plt.close()

    print(f"✅ Gráfico salvo em: {PLOT}")
    print(agrupado)


# ---------------------------
# Execução
# ---------------------------

if __name__ == "__main__":
    df = categorizar_despesas()
    plot_gastos_por_categoria()
