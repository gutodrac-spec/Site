"""GET /api/imoveis -> JSON com todos os imóveis do imoveisdracena.com.br.

A CDN da Vercel guarda a resposta por 5 minutos (s-maxage=300). Depois disso, o
próximo visitante ainda recebe a versão guardada na hora e a Vercel roda o scraper
de novo em segundo plano (stale-while-revalidate). Assim a vitrine se atualiza
sozinha a cada 5 minutos sem deixar nada ligado.
"""
import json, os, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(__file__))
import _scraper as s  # noqa: E402

from bs4 import BeautifulSoup  # noqa: E402


def links_da_pagina(p):
    html = s.baixar(s.LISTA.format(pagina=p))
    if html is None:
        return p, None
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        m = s.LINK_IMOVEL.match(a["href"].strip())
        if m:
            out.append((int(m.group(1)), s.urljoin(s.BASE, a["href"].strip())))
    return p, out


def coletar():
    t0 = time.time()
    links, pagina, lote = {}, 1, 8
    with ThreadPoolExecutor(lote) as ex:
        while pagina < 200:
            fim = False
            for p, achados in sorted(ex.map(links_da_pagina, range(pagina, pagina + lote))):
                if achados is None:
                    raise RuntimeError(f"falha ao ler a listagem, página {p}")
                novos = [x for x in achados if x[0] not in links]
                if not novos:
                    fim = True
                    break
                for i, u in novos:
                    links[i] = u
            if fim:
                break
            pagina += lote

    with ThreadPoolExecutor(16) as ex:
        detalhes = list(ex.map(lambda par: s.detalhar(*par), links.items()))
    imoveis = []
    for d in detalhes:
        if d:
            d["capa"] = d["fotos"][0] if d["fotos"] else None
            imoveis.append(d)
    corpo = json.dumps(imoveis, ensure_ascii=False, sort_keys=True)
    return {
        "fonte": s.BASE,
        "atualizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(imoveis),
        "listados": len(links),
        "segundos": round(time.time() - t0, 1),
        "hash": hashlib.sha1(corpo.encode()).hexdigest(),
        "imoveis": imoveis,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            doc = coletar()
            body = json.dumps(doc, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "public, max-age=0")
            self.send_header("CDN-Cache-Control", "public, s-maxage=300, stale-while-revalidate=86400")
            self.send_header("Vercel-CDN-Cache-Control", "public, s-maxage=300, stale-while-revalidate=86400")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"erro": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
