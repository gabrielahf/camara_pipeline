import os
import time
import requests
from typing import List, Dict, Any

BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"

DEBUG_LIMIT = int(os.getenv("DEBUG_LIMIT", "0"))

def _get(url: str, params: dict = None) -> List[Dict[str, Any]]:
    all_data = []
    page = 1

    while True:
        p = params.copy() if params else {}
        p["pagina"] = page
        p["itens"] = 100

        resp = requests.get(url, params=p, timeout=30)
        resp.raise_for_status()

        data = resp.json().get("dados", [])

        if not data:
            break

        all_data.extend(data)

        if DEBUG_LIMIT and len(all_data) >= DEBUG_LIMIT:
            return all_data[:DEBUG_LIMIT]

        page += 1
        time.sleep(0.1)  

    return all_data


def get_deputados(legislatura: int):
    return _get(f"{BASE_URL}/deputados", {"idLegislatura": legislatura})


def get_despesas_deputado(deputado_id: int, ano: int):
    return _get(
        f"{BASE_URL}/deputados/{deputado_id}/despesas",
        {"ano": ano}
    )


def get_proposicoes_por_autor(deputado_id: int, ano: int):
    return _get(
        f"{BASE_URL}/proposicoes",
        {"idDeputadoAutor": deputado_id, "ano": ano}
    )


def get_eventos_deputado(deputado_id: int, ano: int):
    return _get(
        f"{BASE_URL}/deputados/{deputado_id}/eventos",
        {"dataInicio": f"{ano}-01-01", "dataFim": f"{ano}-12-31"}
    )
