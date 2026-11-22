"""Agenda e baixa as séries históricas da Plataforma de Entrega de Dados (PED) do CEMADEN.

    python scripts/download_ped.py schedule --manifest <novo.csv> --cache <pasta>
    python scripts/download_ped.py download --manifest <novo.csv> --cache <pasta> --poll-minutes 5

Requer conta na PED. As credenciais ficam fora do repositório, em ordem de preferência:

- ``CEMADEN_CREDENTIALS_FILE``: arquivo JSON com ``{"email": ..., "password": ...}``; o token (válido por 4 horas)
  é pedido ao SGAA e renovado automaticamente;
- ``CEMADEN_TOKEN`` ou ``CEMADEN_TOKEN_FILE``: um token já emitido, sem renovação.

Token e credenciais não são gravados em disco. As chamadas respeitam o limite de 12 por minuto para usuários
externos (intervalo mínimo de 6 s) e cada resposta de consulta é guardada em ``--cache``.

``schedule`` agenda um pedido por estado e tipo de estação (5 = Acqua, 4 = Agro) para cada ano civil a partir de
maio de 2019, mais um bloco de 2014 a abril de 2019, e grava o número de cada pedido no manifesto. ``download``
baixa os pedidos concluídos do manifesto para ``data/raw/ped/`` (os links valem 30 dias). Os pedidos usados no estudo
estão em ``data/raw/ped_requests.csv``; seus links já expiraram, então uma nova obtenção exige agendar de novo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://sws.cemaden.gov.br/PED/rest/"
SGAA_TOKENS = "https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens"
MIN_INTERVAL = 6.0                     # segundos entre chamadas
STATES = ("AL", "BA", "CE", "MG", "PB", "PE", "PI", "RN", "SE")
TYPES = (5, 4)                         # Acqua e agrometeorológica
FINAL = ("CONCLUIDA", "REJEITADA")     # estados em que o pedido não muda mais
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ped"


class PED:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = self.cache_dir / "calls.log"
        self.last_call = 0.0
        self.credentials = None
        if path := os.environ.get("CEMADEN_CREDENTIALS_FILE"):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not data.get("email") or not data.get("password"):
                raise RuntimeError("o arquivo de credenciais precisa de 'email' e 'password'")
            self.credentials = {"email": data["email"], "password": data["password"]}
        token = os.environ.get("CEMADEN_TOKEN")
        if not token and os.environ.get("CEMADEN_TOKEN_FILE"):
            token = Path(os.environ["CEMADEN_TOKEN_FILE"]).read_text(encoding="utf-8").strip()
        self.token = token or (self.renew_token() if self.credentials else None)
        if not self.token:
            raise RuntimeError("defina CEMADEN_CREDENTIALS_FILE, CEMADEN_TOKEN ou CEMADEN_TOKEN_FILE")

    def wait(self) -> None:
        delay = MIN_INTERVAL - (time.monotonic() - self.last_call)
        if delay > 0:
            time.sleep(delay)
        self.last_call = time.monotonic()

    def record(self, status: int, what: str, size: int) -> None:
        with self.log.open("a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{status}\t{what}\t{size} bytes\n")

    def renew_token(self) -> str:
        if not self.credentials:
            raise RuntimeError("token expirado e sem arquivo de credenciais para renová-lo")
        self.wait()
        request = urllib.request.Request(SGAA_TOKENS, data=json.dumps(self.credentials).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, body = response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            status, body = error.code, ""
        self.record(status, "SGAA controle-token/tokens", len(body))
        if status != 200:
            raise RuntimeError(f"o SGAA recusou as credenciais (status {status})")
        self.token = json.loads(body)["token"]
        return self.token

    def get(self, path: str, use_cache: bool = True, **params) -> tuple[int, object]:
        params = {k: v for k, v in params.items() if v is not None}
        params.setdefault("formato", "JSON")
        query = urllib.parse.urlencode(params)
        cached = self.cache_dir / f"{path.replace('/', '_')}_{hashlib.sha1(f'{path}?{query}'.encode()).hexdigest()[:16]}.json"
        if use_cache and cached.exists():
            return 200, json.loads(cached.read_text(encoding="utf-8"))
        for attempt in range(2):
            self.wait()
            request = urllib.request.Request(f"{BASE}{path}?{query}",
                                             headers={"token": self.token, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    status, body = response.status, response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read().decode("utf-8", errors="replace")
            self.record(status, f"{path}?{query}", len(body))
            if status in (400, 401) and self.credentials and attempt == 0:   # token expirado ou em branco
                self.renew_token()
                continue
            break
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = body
        if status in (200, 202) and use_cache:
            cached.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return status, data


def yearly_windows(start: str) -> list[tuple[str, str]]:
    """Um intervalo por ano civil de ``start`` (AAAAMMDDhhmm) até hoje e o bloco de 2014 a abril de 2019."""
    first, today = int(start[:4]), time.strftime("%Y%m%d2359")
    out = [(start if y == first else f"{y}01010000", f"{y}12312359" if f"{y}12312359" < today else today)
           for y in range(first, int(today[:4]) + 1)]
    return out + [("201401010000", "201904302359")]


def schedule(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    done = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8") as f:
            done = {(r["uf"], r["tipoestacao"], r["datainicio"], r["datafim"]) for r in csv.DictReader(f) if r["id"]}
    ped = PED(Path(args.cache))
    windows = [tuple(w.split("-")) for w in args.windows] if args.windows else yearly_windows(args.start)
    new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uf", "tipoestacao", "datainicio", "datafim", "status", "id", "agendado_em"])
        if new:
            writer.writeheader()
        for start, end in windows:
            for uf in args.states:
                for kind in args.types:
                    if (uf, str(kind), start, end) in done:
                        continue
                    status, data = ped.get("controle-agendamento/pcds-dados-historicos", use_cache=False,
                                           datainicio=start, datafim=end, rede=11, uf=uf, tipoestacao=kind,
                                           arquivo="JSON")
                    writer.writerow({"uf": uf, "tipoestacao": kind, "datainicio": start, "datafim": end,
                                     "status": status, "id": data.get("id") if isinstance(data, dict) else "",
                                     "agendado_em": time.strftime("%Y-%m-%d %H:%M:%S")})
                    f.flush()
                    print(uf, kind, start, end, status, str(data)[:120], flush=True)
                    if status == 401:
                        print("token expirado ou inválido; interrompido", flush=True)
                        return


def download_ready(ped: PED, requests: dict, out: Path) -> int:
    """Baixa os pedidos prontos do manifesto e devolve quantos seguem pendentes."""
    status, scheduled = ped.get("controle-agendamento/agendamentos", use_cache=False)
    if status != 200 or not isinstance(scheduled, list):
        raise SystemExit(f"falha ao listar os pedidos: {status} {str(scheduled)[:200]}")
    rows = []
    for item in scheduled:
        rid = str(item.get("id"))
        meta = requests.get(rid)
        state = (item.get("status") or {}).get("description")
        link = item.get("link")
        rows.append({"id": rid, "status": state, "link_host": link.split("/")[2] if link and "//" in link else "",
                     **({k: meta[k] for k in ("uf", "tipoestacao", "datainicio", "datafim")} if meta else {})})
        if not (meta and link and state == "CONCLUIDA"):
            continue
        suffix = Path(link.split("?")[0]).suffix or ".dat"
        target = out / f"{rid}_{meta['uf']}_{meta['tipoestacao']}_{meta['datainicio'][:8]}{suffix}"
        if target.exists():
            continue
        time.sleep(MIN_INTERVAL)
        try:
            with urllib.request.urlopen(urllib.request.Request(link), timeout=600) as response:
                target.write_bytes(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                print(f"{rid}: erro {error.code} ao baixar", flush=True)
                continue
            time.sleep(MIN_INTERVAL)       # link protegido: repete com o token
            with urllib.request.urlopen(urllib.request.Request(link, headers={"token": ped.token}), timeout=600) as r:
                target.write_bytes(r.read())
        print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB", flush=True)
    with (out / "status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "link_host", "uf", "tipoestacao", "datainicio", "datafim"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["id"] in requests and r["status"] == "REJEITADA":
            print(f"rejeitado pela PED: {r['id']} {r.get('uf')} tipo {r.get('tipoestacao')} "
                  f"{r.get('datainicio')}–{r.get('datafim')} (agende em intervalos menores com --windows)", flush=True)
    return sum(1 for r in rows if r["id"] in requests and r["status"] not in FINAL)


def download(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        requests = {r["id"]: r for r in csv.DictReader(f) if r["id"]}
    ped = PED(Path(args.cache))
    while (pending := download_ready(ped, requests, out)) and args.poll_minutes:
        print(f"{time.strftime('%H:%M')} {pending} pedidos pendentes", flush=True)
        time.sleep(args.poll_minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("schedule", help="agenda os pedidos e grava o manifesto")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--states", nargs="+", default=list(STATES))
    p.add_argument("--types", nargs="+", type=int, default=list(TYPES))
    p.add_argument("--start", default="201905010000")
    p.add_argument("--windows", nargs="+", help="intervalos explícitos AAAAMMDDhhmm-AAAAMMDDhhmm")
    p.set_defaults(func=schedule)
    p = sub.add_parser("download", help="baixa os pedidos concluídos do manifesto")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--poll-minutes", type=float, help="repete até não restar pedido pendente")
    p.set_defaults(func=download)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
