#!/usr/bin/env python3
"""
Scraper do imoveisdracena.com.br -> imoveis.json

Uso:
  pip install requests beautifulsoup4
  python scraper.py                 # roda uma vez e grava imoveis.json
  python scraper.py --loop 300      # roda a cada 5 minutos (300 s), para sempre
  python scraper.py --saida site/imoveis.json

O que faz:
  1. Percorre a listagem /imobiliaria/imoveis/0/1, /0/2 ... até não aparecer imóvel novo.
  2. Abre a página de cada imóvel e extrai título, preço, código, quartos, suítes,
     banheiros, vagas, áreas, bairro, cidade, descrição completa e todas as fotos.
  3. Grava o JSON só quando algo mudou (o arquivo é trocado de uma vez, nunca fica pela metade).

Em modo --loop, a página de cada imóvel só é baixada de novo quando ele é novo
ou a cada --refazer-a-cada segundos (padrão 6 h). A listagem é lida em toda rodada,
então imóvel vendido/retirado some do JSON em até 5 minutos.
"""
import argparse, hashlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.imoveisdracena.com.br"
LISTA = BASE + "/imobiliaria/imoveis/0/{pagina}"
LINK_IMOVEL = re.compile(r"^(?:https?://(?:www\.)?imoveisdracena\.com\.br)?/(\d+)/imoveis/([a-z0-9\-]+)/?$", re.I)
FOTO = re.compile(r"/exportacao/fotos/[^\"'\s)]+\.(?:jpe?g|png|webp)", re.I)
MINI = re.compile(r"/admin/imovel/mini/([^\"'\s)/]+\.(?:jpe?g|png|webp))", re.I)

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (scraper vitrine imoveisdracena)"


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def baixar(url, tentativas=3):
    for i in range(tentativas):
        try:
            r = S.get(url, timeout=25)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except requests.RequestException as e:
            if i == tentativas - 1:
                log("falhou:", url, e)
                return None
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------- listagem
def listar_links(max_paginas=100):
    """Devolve {id: url} de todos os imóveis da listagem, na ordem do site."""
    achados = {}
    for pagina in range(1, max_paginas + 1):
        html = baixar(LISTA.format(pagina=pagina))
        if html is None:
            raise RuntimeError(f"não consegui ler a página {pagina} da listagem")
        soup = BeautifulSoup(html, "html.parser")
        novos = 0
        for a in soup.find_all("a", href=True):
            m = LINK_IMOVEL.match(a["href"].strip())
            if m and int(m.group(1)) not in achados:
                achados[int(m.group(1))] = urljoin(BASE, a["href"].strip())
                novos += 1
        log(f"listagem página {pagina}: {novos} novos (total {len(achados)})")
        if novos == 0:
            break
        time.sleep(0.4)
    return achados


# ---------------------------------------------------------------- detalhe
def numero(txt):
    if txt is None:
        return None
    txt = txt.strip().replace(".", "").replace(",", ".")
    try:
        v = float(txt)
        return int(v) if v.is_integer() else v
    except ValueError:
        return None


def campo(texto, *rotulos, unidade=""):
    """Acha 'Rótulo: 3' ou '3 Rótulo' no texto corrido da página."""
    num = r"(\d[\d\.]*(?:,\d+)?)"
    for r in rotulos:
        for padrao in (rf"{r}\s*:?\s*{num}\s*{unidade}", rf"{num}\s*{unidade}\s*{r}"):
            m = re.search(padrao, texto, re.I)
            if m:
                return numero(m.group(1))
    return None


def texto_rotulo(texto, rotulo):
    m = re.search(rf"{rotulo}\s*:?\s*([^\n]+)", texto, re.I)
    return m.group(1).strip() if m else None


def meta(soup, prop):
    t = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
    return t.get("content", "").strip() if t else None


def extrair_descricao(soup):
    """Pega o bloco que contém 'Descrição:' (texto completo, não o og:description cortado)."""
    alvo = soup.find(string=re.compile(r"Descri[cç][aã]o\s*:", re.I))
    if alvo:
        el = alvo.parent
        # sobe até um bloco com texto suficiente
        while el is not None and len(el.get_text(" ", strip=True)) < 120 and el.parent is not None:
            el = el.parent
        if el is not None:
            linhas = [l.strip() for l in el.get_text("\n").splitlines()]
            txt = "\n".join(l for l in linhas if l)
            return re.sub(r"\n{3,}", "\n\n", txt)
    return meta(soup, "og:description")


def extrair_fotos(html, soup):
    fotos = []
    def add(u):
        u = urljoin(BASE, u)
        if u not in fotos:
            fotos.append(u)
    # capa: o og:image fica em /admin/imovel/mini/; a versão grande tem o mesmo nome em /exportacao/fotos/
    og = meta(soup, "og:image") or ""
    m = MINI.search(og)
    if m:
        add("/exportacao/fotos/" + m.group(1))
    for m in FOTO.finditer(html):
        add(m.group(0))
    if not fotos and og:
        add(og)
    return fotos


def detalhar(iid, url):
    html = baixar(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    texto = soup.get_text("\n")
    texto_plano = re.sub(r"[ \t]+", " ", texto)

    h1 = soup.find("h1")
    titulo = (h1.get_text(" ", strip=True) if h1 else None) or meta(soup, "og:title")
    preco = re.search(r"R\$\s*([\d\.]+(?:,\d{2})?)", texto_plano)
    codigo = re.search(r"(?:C[óo]d\.?|REF:?)\s*([A-Za-z0-9\-]+)", texto_plano)
    cidade = texto_rotulo(texto_plano, r"Cidade")
    local = texto_rotulo(texto_plano, r"Localiza[cç][aã]o")

    slug = url.rstrip("/").split("/")[-1]
    tipo = re.match(r"(venda|locacao|aluguel)-([a-z\-]+?)(?:-\d+-quartos?|-[a-z]|$)", slug)

    return {
        "id": iid,
        "url": url,
        "codigo": codigo.group(1) if codigo else None,
        "titulo": titulo,
        "transacao": "locação" if "/locacao-" in url or "/aluguel-" in url else "venda",
        "tipo": tipo.group(2).replace("-", " ").strip() if tipo else None,
        "preco": numero(preco.group(1)) if preco else None,
        "quartos": campo(texto_plano, r"Quartos?", r"Dormit[óo]rios?"),
        "suites": campo(texto_plano, r"Su[íi]tes?"),
        "banheiros": campo(texto_plano, r"Banheiros?"),
        "vagas": campo(texto_plano, r"Vagas?"),
        "area_construida": campo(texto_plano, r"[ÁA]rea constru[íi]da", unidade=r"m²?"),
        "area_terreno": campo(texto_plano, r"[ÁA]rea do terreno", r"[ÁA]rea total", unidade=r"m²?"),
        "bairro": (local.split(" - ")[0].strip() if local else None),
        "cidade": cidade,
        "descricao": extrair_descricao(soup),
        "fotos": extrair_fotos(html, soup),
        "capa": None,  # preenchido abaixo
    }


# ---------------------------------------------------------------- rodada
def rodada(saida, cache, refazer_a_cada, trabalhadores):
    links = listar_links()
    agora = time.time()
    pendentes = [(i, u) for i, u in links.items()
                 if i not in cache or agora - cache[i]["_lido"] > refazer_a_cada]
    log(f"{len(links)} imóveis no site, {len(pendentes)} para (re)ler")

    def tarefa(par):
        d = detalhar(*par)
        time.sleep(0.3)
        return par[0], d

    with ThreadPoolExecutor(trabalhadores) as ex:
        for iid, d in ex.map(tarefa, pendentes):
            if d:
                d["capa"] = d["fotos"][0] if d["fotos"] else None
                d["_lido"] = agora
                cache[iid] = d

    for iid in list(cache):          # imóvel saiu do site -> sai do JSON
        if iid not in links:
            del cache[iid]

    imoveis = [{k: v for k, v in cache[i].items() if k != "_lido"} for i in links if i in cache]
    corpo = json.dumps(imoveis, ensure_ascii=False, sort_keys=True)
    hash_novo = hashlib.sha1(corpo.encode()).hexdigest()

    hash_antigo = None
    if os.path.exists(saida):
        try:
            with open(saida, encoding="utf-8") as f:
                hash_antigo = json.load(f).get("hash")
        except Exception:
            pass
    if hash_novo == hash_antigo:
        log("nada mudou, JSON mantido")
        return

    doc = {
        "fonte": BASE,
        "atualizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(imoveis),
        "hash": hash_novo,
        "imoveis": imoveis,
    }
    pasta = os.path.dirname(os.path.abspath(saida))
    os.makedirs(pasta, exist_ok=True)
    tmp = saida + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, saida)
    log(f"gravado {saida}: {len(imoveis)} imóveis")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="imoveis.json")
    ap.add_argument("--loop", type=int, default=0, help="segundos entre rodadas (ex.: 300 = 5 min); 0 roda uma vez")
    ap.add_argument("--refazer-a-cada", type=int, default=6 * 3600, help="segundos até reler a página de um imóvel já conhecido")
    ap.add_argument("--trabalhadores", type=int, default=4)
    a = ap.parse_args()

    cache = {}
    while True:
        try:
            rodada(a.saida, cache, a.refazer_a_cada, a.trabalhadores)
        except Exception as e:
            log("erro na rodada:", e)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
