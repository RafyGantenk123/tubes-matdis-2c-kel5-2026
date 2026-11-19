"""Test offline untuk scripts/fetch_distances.py (tanpa internet / API key)."""
import networkx as nx
import pytest
import yaml

from cases.scenario_loader import ScenarioLoader
from scripts import fetch_distances as fd


def make_cfg(n=8, knn=3):
    return {
        "settings": {"case_id": "case_test", "title": "Uji", "tarif_per_ton_km": 5,
                     "bongkar_muat": 20, "knn": knn, "start_node": 0, "goal_node": n - 1},
        "nodes": [{"id": i, "label": f"T{i}", "terrain": 1.0 if i < n // 2 else 2.0,
                   "pos": [-2.5, 140.0 + 0.1 * i]} for i in range(n)],
    }


def line_matrix(n, km_step=10):
    """Titik berjajar di satu jalan lurus: jarak i-j = |i-j| x km_step."""
    dist = [[abs(i - j) * km_step * 1000.0 for j in range(n)] for i in range(n)]
    dur = [[abs(i - j) * 600.0 for j in range(n)] for i in range(n)]
    return dist, dur


def test_weight_formula():
    cfg = make_cfg(2, knn=1)
    dist, dur = line_matrix(2)  # 10 km, medan max(1.0, 2.0) = 2.0
    sc = fd.build_case(cfg, dist, dur)
    # 10 km x 5 x 2.0 + 20 = 120
    assert sc["edges"][0]["weight"] == 120.0
    assert sc["edges"][0]["terrain"] == 2.0
    assert sc["edges"][0]["duration_min"] == 10.0


def test_yaml_loads_with_template_loader_and_mst_has_v_minus_1_edges(tmp_path):
    cfg = make_cfg(8)
    sc = fd.build_case(cfg, *line_matrix(8))
    out = tmp_path / "case.yaml"
    out.write_text(yaml.safe_dump(sc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    model = ScenarioLoader(out).build_graph_model()
    g = nx.Graph(model.nx_graph)
    assert len(g.nodes) == 8 and nx.is_connected(g)
    assert len(nx.minimum_spanning_tree(g).edges) == 7
    lon, lat = model.nx_graph.nodes[0]["pos"]
    assert lon > 100 and lat < 0  # pos = [lon, lat]


def test_pairs_without_route_are_skipped_but_graph_stays_connected():
    cfg = make_cfg(5, knn=4)
    dist, dur = line_matrix(5)
    dist[0][4] = dist[4][0] = None
    sc = fd.build_case(cfg, dist, dur)
    assert {0, 4} not in [{e["from"], e["to"]} for e in sc["edges"]]


def test_one_direction_only_is_still_used():
    cfg = make_cfg(2, knn=1)
    dist, dur = line_matrix(2)
    dist[1][0] = None  # hanya arah 0 -> 1 yang ada
    assert len(fd.build_case(cfg, dist, dur)["edges"]) == 1


def test_disconnected_network_raises():
    cfg = make_cfg(4, knn=3)
    dist, dur = line_matrix(4)
    for i in (0, 1):
        for j in (2, 3):
            dist[i][j] = dist[j][i] = None
    with pytest.raises(ValueError, match="terpecah"):
        fd.build_case(cfg, dist, dur)


def test_ors_request_uses_lon_lat_order(monkeypatch):
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *a): return b'{"distances": [[0]], "durations": [[0]]}'

    def fake_urlopen(req, timeout=0):
        seen["body"] = __import__("json").loads(req.data)
        seen["auth"] = req.get_header("Authorization")
        return FakeResp()

    monkeypatch.setattr(fd.urllib.request, "urlopen", fake_urlopen)
    fd.fetch_matrix([(-2.5, 140.7)], "KEY123")
    assert seen["body"]["locations"] == [[140.7, -2.5]]
    assert seen["auth"] == "KEY123"


def test_shipped_nodes_yaml_has_20_plus_unique_nodes():
    cfg = yaml.safe_load(fd.NODES_FILE.read_text(encoding="utf-8"))
    ids = [n["id"] for n in cfg["nodes"]]
    assert len(ids) >= 20 and len(set(ids)) == len(ids)
    assert cfg["settings"]["goal_node"] in ids