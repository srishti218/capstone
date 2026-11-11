import pandas as pd
import pytest


def test_ring_component_links_distinct_accounts_without_labels():
    pytest.importorskip("torch_geometric")
    from src.graph.rings import build_entity_graph

    rows = pd.DataFrame({
        "account_id": ["a", "b", "c"],
        "card_id": ["one", "two", "three"],
        "device_id": ["shared", "shared", "other"],
        "TransactionAmt": [1.0, 2.0, 3.0],
    })
    indices, rings, features, edges = build_entity_graph(rows)
    assert rings["account:id:a"] == rings["account:id:b"] == rings["device:id:shared"]
    assert "account:id:c" not in rings
    assert len(indices) == 8 and features.shape == (9, 6)
    edge_set = set(map(tuple, edges.T.tolist()))
    assert (indices["card:id:one"], indices["device:id:shared"]) in edge_set
    assert all((target, source) in edge_set for source, target in edge_set)
