"""API tests: happy paths, validation errors, health, and metrics.

All run offline against the stub classifier via the ``client`` fixture.
"""
from __future__ import annotations

from app import LABELS


# --- /predict happy paths ---------------------------------------------------

def test_predict_single_happy_path(client):
    r = client.post("/predict", json={"text": "i am so happy and grateful today"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in LABELS
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["probabilities"]) == set(LABELS)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
    # score must equal the probability of the reported top label
    assert abs(body["score"] - body["probabilities"][body["label"]]) < 1e-9


def test_predict_single_label_matches_lexicon(client):
    r = client.post("/predict", json={"text": "i am terrified, i am so scared"})
    assert r.status_code == 200
    assert r.json()["label"] == "fear"


def test_predict_batch_happy_path(client):
    texts = ["i feel wonderful", "i am furious", "i adore you"]
    r = client.post("/predict", json={"texts": texts})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == len(texts)
    labels = [p["label"] for p in preds]
    assert labels == ["joy", "anger", "love"]
    for p in preds:
        assert set(p["probabilities"]) == set(LABELS)


# --- /predict validation errors --------------------------------------------

def test_predict_empty_text_is_422(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_blank_text_is_422(client):
    r = client.post("/predict", json={"text": "    "})
    assert r.status_code == 422


def test_predict_missing_fields_is_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_both_fields_is_422(client):
    r = client.post("/predict", json={"text": "hi", "texts": ["hi"]})
    assert r.status_code == 422


def test_predict_empty_batch_is_422(client):
    r = client.post("/predict", json={"texts": []})
    assert r.status_code == 422


def test_predict_batch_with_blank_item_is_422(client):
    r = client.post("/predict", json={"texts": ["ok", "   "]})
    assert r.status_code == 422


def test_predict_oversized_batch_is_422(client):
    # default MAX_BATCH_SIZE is 64
    r = client.post("/predict", json={"texts": ["x"] * 65})
    assert r.status_code == 422


def test_predict_wrong_type_is_422(client):
    r = client.post("/predict", json={"text": 12345})
    assert r.status_code == 422


# --- /healthz ---------------------------------------------------------------

def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"
    assert body["offline"] is True
    assert body["labels"] == LABELS
    assert "version" in body


# --- /metrics ---------------------------------------------------------------

def test_metrics_exposes_prometheus(client):
    # generate some traffic first so counters are non-zero
    client.post("/predict", json={"text": "hello world"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    text = r.text
    assert "emotion_api_requests_total" in text
    assert "emotion_api_request_latency_seconds" in text
    assert "emotion_api_inference_latency_seconds" in text


# --- root + demo ------------------------------------------------------------

def test_root_redirects_to_demo(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/demo"


def test_demo_page_served(client):
    r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Emotion" in r.text
