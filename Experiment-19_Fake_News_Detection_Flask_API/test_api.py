"""API tests.  Run:  python -m pytest test_api.py -v"""

import pytest

from app import app

REAL = ("The finance ministry said on Tuesday that the national budget will be reviewed next month, "
        "according to Reuters. A committee has been formed to study the proposal.")
FAKE = ("SHOCKING!!! The health minister secretly admits the vaccine programme is a total SCAM. "
        "Share before they delete this! Mainstream media is silent!!!")


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/api/health").json["status"] == "ok"


def test_real_article(client):
    assert client.post("/api/predict", json={"text": REAL}).json["label"] == "REAL"


def test_fake_article(client):
    assert client.post("/api/predict", json={"text": FAKE}).json["label"] == "FAKE"


def test_batch(client):
    labels = [r["label"] for r in client.post("/api/batch", json={"articles": [REAL, FAKE]}).json["results"]]
    assert labels == ["REAL", "FAKE"]


def test_short_text_rejected(client):
    assert client.post("/api/predict", json={"text": "hi"}).status_code == 400
