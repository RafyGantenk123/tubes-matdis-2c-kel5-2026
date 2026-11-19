"""Ambil jarak jalan dari OpenRouteService (ORS) -> skenario YAML untuk MST.

Langkah:
    1. Daftar gratis di openrouteservice.org, buat API key.
    2. Simpan di file .env :   ORS_API_KEY=isi_key_anda
    3. Isi koordinat tiap titik di data/nodes.yaml  (pos: [lat, lon])
    4. Jalankan:               python -m scripts.fetch_distances

Hasil: cases/scenarios/advanced/case_mst_papua.yaml
Matriks mentah disimpan di data/ors_matrix.json, jadi menjalankan ulang
tidak memanggil API lagi (pakai --refresh untuk memaksa panggil ulang).
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

import networkx as nx
import yaml

from config.settings import BASE_DIR, CASES_DIR

NODES_FILE = BASE_DIR / "data" / "nodes.yaml"
CACHE_FILE = BASE_DIR / "data" / "ors_matrix.json"
OUT_FILE = CASES_DIR / "advanced" / "case_mst_papua.yaml"  # di luar glob test_all_scenarios
ORS_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"  # coba driving-hgv untuk truk


def get_api_key():
    if os.environ.get("ORS_API_KEY"):
        return os.environ["ORS_API_KEY"]
    env = BASE_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ORS_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit("ORS_API_KEY belum diisi (file .env atau environment variable).")


def fetch_matrix(coords, key):
    """Satu request untuk semua pasangan titik. coords = [(lat, lon), ...]."""
    body = {
        "locations": [[lon, lat] for lat, lon in coords],  # ORS memakai urutan lon, lat
        "metrics": ["distance", "duration"],
        "units": "m",
    }
    req = urllib.request.Request(
        ORS_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"ORS menolak request (HTTP {e.code}): {e.read().decode()[:500]}")


def straight_km(a, b):
    """Jarak garis lurus (km), hanya untuk memeriksa hasil, bukan untuk bobot."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def build_case(cfg, dist_m, dur_s):
    """Matriks jarak -> dict skenario. Bobot = jarak_km x tarif x faktor_medan + bongkar_muat."""
    s, nodes = cfg["settings"], cfg["nodes"]
    n = len(nodes)
    cands = {}  # (i, j) -> info edge, hanya pasangan yang punya rute jalan
    for i in range(n):
        for j in range(i + 1, n):
            both = [v for v in (dist_m[i][j], dist_m[j][i]) if v is not None]
            if not both:
                continue
            km = sum(both) / len(both) / 1000
            terrain = max(nodes[i]["terrain"], nodes[j]["terrain"])
            mins = [v for v in (dur_s[i][j], dur_s[j][i]) if v is not None]
            cands[(i, j)] = {
                "weight": round(km * s["tarif_per_ton_km"] * terrain + s["bongkar_muat"], 1),
                "distance_km": round(km, 2),
                "duration_min": round(sum(mins) / len(mins) / 60, 1) if mins else None,
                "terrain": terrain,
            }

    # simpan k tetangga termurah tiap titik (graf lengkap 210 edge terlalu padat)
    chosen = set()
    for i in range(n):
        mine = sorted((c["weight"], pair) for pair, c in cands.items() if i in pair)
        chosen.update(pair for _, pair in mine[: s["knn"]])

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(chosen)
    if not nx.is_connected(g):
        parts = [sorted(nodes[i]["label"] for i in c) for c in nx.connected_components(g)]
        raise ValueError(f"Graf terpecah jadi {len(parts)} kelompok: {parts}. Naikkan `knn` atau periksa koordinat.")

    return {
        "case_id": s["case_id"],
        "title": s["title"],
        "description": f"MST jaringan logistik material, {n} titik, {len(chosen)} jalur. Jarak jalan dari OpenRouteService.",
        "nodes": [{"id": nd["id"], "label": nd["label"], "pos": [nd["pos"][1], nd["pos"][0]]} for nd in nodes],  # [lon, lat]
        "edges": [{"from": nodes[i]["id"], "to": nodes[j]["id"], **cands[(i, j)]} for i, j in sorted(chosen)],
        "execution_plan": {
            "algorithm": "spanning_tree",
            "start_node": s["start_node"],
            "goal_node": s["goal_node"],
            "events": [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="panggil ORS lagi, abaikan cache")
    args = parser.parse_args()

    cfg = yaml.safe_load(NODES_FILE.read_text(encoding="utf-8"))
    missing = [n["label"] for n in cfg["nodes"] if not n.get("pos")]
    if missing:
        sys.exit("Koordinat (pos) belum diisi di data/nodes.yaml untuk: " + ", ".join(missing))
    coords = [tuple(n["pos"]) for n in cfg["nodes"]]

    cached = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else None
    if cached and not args.refresh and cached["coords"] == [list(c) for c in coords]:
        print("Memakai data/ors_matrix.json (tanpa panggilan API)")
        dist, dur = cached["distances"], cached["durations"]
    else:
        resp = fetch_matrix(coords, get_api_key())
        dist, dur = resp["distances"], resp["durations"]
        CACHE_FILE.write_text(json.dumps({"coords": coords, "distances": dist, "durations": dur}))

    # pemeriksaan cepat: jalan jauh lebih panjang dari garis lurus = curiga
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if dist[i][j] is None:
                print(f"[TANPA RUTE] {cfg['nodes'][i]['label']} - {cfg['nodes'][j]['label']}")
            elif dist[i][j] / 1000 > 2.5 * max(straight_km(coords[i], coords[j]), 1):
                print(f"[CURIGA] {cfg['nodes'][i]['label']} - {cfg['nodes'][j]['label']}: "
                      f"jalan {dist[i][j] / 1000:.0f} km vs lurus {straight_km(coords[i], coords[j]):.0f} km")

    try:
        scenario = build_case(cfg, dist, dur)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True), encoding="utf-8")

    g = nx.Graph([(e["from"], e["to"], {"weight": e["weight"]}) for e in scenario["edges"]])
    total = sum(d["weight"] for *_, d in g.edges(data=True))
    mst = nx.minimum_spanning_tree(g)
    mst_total = sum(d["weight"] for *_, d in mst.edges(data=True))
    print(f"\nTersimpan: {OUT_FILE}")
    print(f"{g.number_of_nodes()} titik | {g.number_of_edges()} edge | total bobot {total:,.1f}")
    print(f"MST: {mst.number_of_edges()} edge | total bobot {mst_total:,.1f}")


if __name__ == "__main__":
    main()