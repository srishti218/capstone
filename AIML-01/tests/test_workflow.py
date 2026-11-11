from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from kafka import TopicPartition

from src.api import main
from src.api.schemas import FraudScore
from src.streaming.consumer import process_message
from src.streaming.producer import transactions


def test_score_retry_and_case_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_STATE_DIR", str(tmp_path))
    model = Mock()
    model.predict_proba.return_value = np.array([[0.15, 0.85]])
    monkeypatch.setattr(main, "model", model)
    monkeypatch.setattr(main, "feature_columns", ["TransactionAmt"])
    monkeypatch.setattr(main, "scoring_context", lambda *_: {})
    monkeypatch.setattr(main, "RING_PATH", tmp_path / "absent.pt")

    with TestClient(main.app) as client:
        event = {"transaction_id": "case/one", "TransactionAmt": 200, "TransactionDT": 10}
        scored = client.post("/score", json=event)
        assert scored.json()["decision"] == "REVIEW"
        assert client.post("/score", json=event).json() == scored.json()
        assert client.post("/score", json={**event, "TransactionAmt": 201}).status_code == 409
        assert model.predict_proba.call_count == 1
        assert client.get("/cases").json()["counts"] == {"open": 1}
        assert client.patch("/cases/case/one", json={"status": "closed"}).status_code == 422
        response = client.patch("/cases/case/one", json={"status": "closed", "label": "fraud"})
        assert response.json()["label"] == "fraud"
        assert client.get("/cases?status=open").json()["cases"] == []
        assert client.get("/cases?status=closed").json()["counts"] == {"closed": 1}


def test_replay_is_ordered_and_excludes_ground_truth(tmp_path):
    path = tmp_path / "transactions.csv"
    path.write_text("TransactionID,TransactionDT,TransactionAmt,card1,isFraud,ring_label\n"
                    "2,20,300,9,1,1\n1,10,12,8,0,0\n")
    records = list(transactions(path, start_time=0, limit=2))
    assert [record["transaction_id"] for record in records] == ["replay-1", "replay-2"]
    assert all(not record["extra_features"] for record in records)


def test_consumer_commits_only_after_publish_acknowledgement():
    message = SimpleNamespace(
        topic="transactions", partition=2, offset=15, key=b"replay-1",
        value=b'{"transaction_id":"replay-1","TransactionDT":1,"TransactionAmt":12}',
    )
    producer, consumer = Mock(), Mock()
    score = Mock(return_value=FraudScore(transaction_id="replay-1", fraud_probability=0.1,
                                       risk_score=10, decision="ALLOW", model="xgboost-baseline"))
    ack = producer.send.return_value.get
    ack.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        process_message(message, score, producer, consumer)
    consumer.commit.assert_not_called()
    ack.side_effect = None
    process_message(message, score, producer, consumer)
    offset = consumer.commit.call_args.args[0][TopicPartition("transactions", 2)]
    assert offset.offset == 16
    assert producer.send.call_args.args == ("fraud-scores",)
