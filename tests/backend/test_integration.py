"""End-to-end integration tests using TestClient and a 1000-row CSV."""
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def make_1000_row_csv(path: str) -> None:
    """Write a 1000-row realistic H-Walker CSV to *path*."""
    rng = np.random.default_rng(7)
    n = 1000
    t = np.linspace(0, n / 111.0, n)
    gait_freq = 1.0

    L_GCP = (t * gait_freq * 100) % 100
    R_GCP = ((t * gait_freq + 0.5) * 100) % 100

    L_ActForce_N = np.clip(40 * np.abs(np.sin(np.pi * t)) + rng.normal(0, 2, n), 0, 70)
    R_ActForce_N = np.clip(38 * np.abs(np.sin(np.pi * (t + 0.5))) + rng.normal(0, 2, n), 0, 70)
    L_DesForce_N = L_ActForce_N + rng.normal(0, 1, n)
    R_DesForce_N = R_ActForce_N + rng.normal(0, 1, n)
    L_Pitch = 5 * np.sin(2 * np.pi * t) + rng.normal(0, 0.5, n)
    R_Pitch = 5 * np.sin(2 * np.pi * (t + 0.5)) + rng.normal(0, 0.5, n)
    L_Roll  = 2 * np.sin(2 * np.pi * t * 2) + rng.normal(0, 0.3, n)
    R_Roll  = 2 * np.sin(2 * np.pi * (t * 2 + 0.5)) + rng.normal(0, 0.3, n)
    L_Yaw   = rng.normal(0, 0.5, n)
    R_Yaw   = rng.normal(0, 0.5, n)
    L_Phase = (L_GCP / 25).astype(int) % 4
    R_Phase = (R_GCP / 25).astype(int) % 4
    L_Event = (np.diff(L_GCP, prepend=L_GCP[0]) < -50).astype(int)
    R_Event = (np.diff(R_GCP, prepend=R_GCP[0]) < -50).astype(int)

    df = pd.DataFrame({
        "timestamp": t,
        "L_ActForce_N": L_ActForce_N,
        "R_ActForce_N": R_ActForce_N,
        "L_DesForce_N": L_DesForce_N,
        "R_DesForce_N": R_DesForce_N,
        "L_GCP": L_GCP,
        "R_GCP": R_GCP,
        "L_Pitch": L_Pitch,
        "R_Pitch": R_Pitch,
        "L_Roll": L_Roll,
        "R_Roll": R_Roll,
        "L_Yaw": L_Yaw,
        "R_Yaw": R_Yaw,
        "L_Phase": L_Phase,
        "R_Phase": R_Phase,
        "L_Event": L_Event,
        "R_Event": R_Event,
    })
    df.to_csv(path, index=False)


@pytest.fixture
def large_csv(tmp_path) -> str:
    path = str(tmp_path / "integration_trial.csv")
    make_1000_row_csv(path)
    return path


class TestIntegrationQuick:
    def test_force_quick_end_to_end(self, large_csv):
        """Full pipeline: POST → analysis_engine → graph_quick → JSON."""
        payload = {
            "request": {
                "analysis_type": "force",
                "columns": ["L_ActForce_N", "R_ActForce_N"],
                "normalize_gcp": False,
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/quick", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        assert "data" in body
        assert "layout" in body
        assert len(body["data"]) == 2  # one trace per column

        trace = body["data"][0]
        assert len(trace["x"]) == 1000
        assert len(trace["y"]) == 1000
        assert trace["mode"] == "lines"

    def test_force_quick_gcp_normalized(self, large_csv):
        payload = {
            "request": {
                "analysis_type": "force",
                "columns": ["L_ActForce_N"],
                "normalize_gcp": True,
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/quick", json=payload)
        assert resp.status_code == 200
        trace = resp.json()["data"][0]
        assert len(trace["x"]) == 101
        assert len(trace["y"]) == 101

    def test_imu_quick_end_to_end(self, large_csv):
        payload = {
            "request": {
                "analysis_type": "imu",
                "sides": ["left"],
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/quick", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        # Left IMU: Pitch, Roll, Yaw = 3 columns
        assert len(body["data"]) == 3
        names = [t["name"] for t in body["data"]]
        assert any("L_Pitch" in n for n in names)
        assert not any("R_" in n for n in names)

    def test_response_is_json_serialisable(self, large_csv):
        payload = {
            "request": {"analysis_type": "force"},
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/quick", json=payload)
        # Will raise if any numpy types leaked through
        json.loads(resp.text)

    def test_journal_list_integration(self):
        resp = client.get("/api/journal/list")
        assert resp.status_code == 200
        keys = {entry["key"] for entry in resp.json()["journals"]}
        assert "ieee_tnsre" in keys
        assert "nature" in keys


class TestIntegrationPublication:
    def test_force_publication_end_to_end(self, large_csv):
        """Full pipeline: POST → analysis_engine → render_svg → SVG bytes."""
        payload = {
            "request": {
                "analysis_type": "force",
                "columns": ["L_ActForce_N", "R_ActForce_N"],
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/publication", json=payload)
        assert resp.status_code == 200
        assert "svg" in resp.headers["content-type"]
        assert "<svg" in resp.text
        assert len(resp.text) > 2000  # real figure has substantial SVG

    def test_publication_ieee_tnsre_journal(self, large_csv):
        payload = {
            "request": {
                "analysis_type": "force",
                "columns": ["L_ActForce_N"],
            },
            "csv_paths": [large_csv],
        }
        resp = client.post(
            "/api/graph/publication?journal=ieee_tnsre",
            json=payload,
        )
        assert resp.status_code == 200
        assert "<svg" in resp.text

    def test_publication_gcp_normalized(self, large_csv):
        payload = {
            "request": {
                "analysis_type": "force",
                "columns": ["L_ActForce_N"],
                "normalize_gcp": True,
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/publication", json=payload)
        assert resp.status_code == 200
        assert "<svg" in resp.text

    def test_publication_imu_left_only(self, large_csv):
        payload = {
            "request": {
                "analysis_type": "imu",
                "sides": ["left"],
            },
            "csv_paths": [large_csv],
        }
        resp = client.post("/api/graph/publication", json=payload)
        assert resp.status_code == 200
        assert "<svg" in resp.text
