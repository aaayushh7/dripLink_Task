from fastapi.testclient import TestClient
import main
import os

client = TestClient(main.app)

def test_detect_sample():
    sample = os.path.join("samples", "en_sample.wav")
    with open(sample, "rb") as f:
        r = client.post("/detect/language", files={"file": ("en_sample.wav", f, "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert "ensemble" in j 