import os
import time
import requests
from typing import List, Dict, Any

BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"

DEBUG_LIMIT = int(os.getenv("DEBUG_LIMIT", "0"))

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def _get(url: str, params: dict = None) -> List[Dict[str, Any]]:
    all_data = []
    page = 1
    
    while True:
        p = params.copy() if params else {}
        p["pagina"] = page
        p["itens"] = 100
        
        # BULLETPROOF RETRY (10 attempts, aggressive backoff)
        for attempt in range(10):
            try:
                resp = requests.get(url, params=p, timeout=120)  # 2min timeout
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt + (page * 0.5)  # Longer for deeper pages
                print(f"Page {page} attempt {attempt+1}: {e} → wait {wait:.1f}s")
                if attempt == 9:
                    print(f"ABANDON page {page} after 10 tries")
                    return all_data  # Return partial data
                time.sleep(wait)
        
        data = resp.json().get("dados", [])
        if not data:
            print(f"Page {page}: empty → done")
            break
            
        all_data.extend(data)
        print(f"Page {page}: +{len(data)} → total {len(all_data)}")
        
        if DEBUG_LIMIT and len(all_data) >= DEBUG_LIMIT:
            print(f"Hit DEBUG_LIMIT={DEBUG_LIMIT}")
            return all_data
        
        page += 1
        time.sleep(2.0)  # Conservative between pages
    return all_data


def get_deputados(legislatura: int):
    data = _get(f"{BASE_URL}/deputados", {"idLegislatura": legislatura})
    return data


def get_despesas_deputado(deputado_id: int):
    data = _get(f"{BASE_URL}/deputados/{deputado_id}/despesas")
    return data


def get_proposicoes_por_autor(deputado_id: int, ano: int):
    data = _get(f"{BASE_URL}/proposicoes", {"idDeputadoAutor": deputado_id, "ano": ano})
    return data


def get_eventos_deputado(deputado_id: int, ano: int):
    data = _get(
        f"{BASE_URL}/deputados/{deputado_id}/eventos",
        {"dataInicio": f"{ano}-01-01", "dataFim": f"{ano}-12-31"},
    )
    return data


import pandas as pd

if __name__ == "__main__":
    Llegislatura = [56, 57]  # To be changed to be read from config
    deps_all = []
    for i in Llegislatura:
        deps = get_deputados(i)
        deps_all.extend(deps)
    df_deps = pd.DataFrame(deps_all)
    deputados = df_deps["id"].unique()
    print("Saved deputados.parquet")
    df_deps.to_parquet(DATA_RAW / "deputados1.parquet", index=False)

    desp_deputados = []
    for dep_id in deputados:
        desp = get_despesas_deputado(dep_id)
        if desp == []:
            continue
        else:
            desp_deputados.extend(desp)
    df_desp = pd.DataFrame(desp_deputados)

    df_desp.to_parquet(DATA_RAW / "despesas1.parquet", index=False)
    print("Saved despesas.parquet")
