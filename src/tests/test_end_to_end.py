import os
import pytest
from fastapi.testclient import TestClient
from main import app
from connectors.whisper_connector import INDIAN_LANGUAGES

client = TestClient(app)

# Test data directory
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")

def test_detect_english():
    """Test English language detection"""
    audio_path = os.path.join(TEST_DIR, "test.wav")
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": audio_path,
            "ground_truth_language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "results" in data
    assert "ensemble" in data
    assert "ground_truth_language" in data
    
    # Check providers
    providers = {r["provider"] for r in data["results"]}
    assert "whisper_local" in providers
    assert "fasttext_local" in providers
    assert "sarvam_mock" in providers
    assert "eleven_mock" in providers
    
    # Check metrics
    for result in data["results"]:
        assert "language" in result
        assert "confidence" in result
        assert "time_taken" in result
        assert "status" in result
        assert "cost" in result
        assert isinstance(result["cost"], dict)
        assert "tokens" in result["cost"]
        assert "usd" in result["cost"]

@pytest.mark.parametrize("lang_code", INDIAN_LANGUAGES.keys())
def test_indian_language_support(lang_code):
    """Test support for various Indian languages"""
    audio_path = os.path.join(TEST_DIR, f"test_{lang_code}.wav")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file for {lang_code} not found")
    
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": audio_path,
            "ground_truth_language": lang_code
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check Whisper detection
    whisper_result = next(r for r in data["results"] if r["provider"] == "whisper_local")
    assert whisper_result["status"] == "success"
    assert whisper_result["confidence"] > 0.35  # Lower threshold for Indian languages
    
    # Check script information
    if whisper_result["language"] in INDIAN_LANGUAGES:
        assert "script" in whisper_result
        assert whisper_result["script"] == INDIAN_LANGUAGES[whisper_result["language"]]
    
    # Check ensemble decision
    assert data["ensemble"]["final_language"] in INDIAN_LANGUAGES

def test_detect_hindi():
    """Test Hindi language detection"""
    audio_path = os.path.join(TEST_DIR, "test_hindi.wav")
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": audio_path,
            "ground_truth_language": "hi"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check Hindi detection
    whisper_result = next(r for r in data["results"] if r["provider"] == "whisper_local")
    assert whisper_result["language"] == "hi"
    assert whisper_result["confidence"] > 0.35  
    assert "script" in whisper_result
    assert whisper_result["script"] == "Devanagari"
    
    # Check ensemble decision
    assert data["ensemble"]["final_language"] == "hi"

def test_invalid_file():
    """Test error handling for invalid file"""
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": "nonexistent.wav",
            "ground_truth_language": "en"
        }
    )
    assert response.status_code == 400
    assert "file not found" in response.json()["detail"].lower()

def test_cost_tracking():
    """Test cost tracking functionality"""
    audio_path = os.path.join(TEST_DIR, "test.wav")
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": audio_path,
            "ground_truth_language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check total cost calculation
    total_cost = data["ensemble"]["total_cost"]
    assert "tokens" in total_cost
    assert "usd" in total_cost
    
    # Verify total cost matches sum of individual costs
    calculated_tokens = sum(r["cost"]["tokens"] for r in data["results"])
    calculated_usd = sum(r["cost"]["usd"] for r in data["results"])
    assert total_cost["tokens"] == calculated_tokens
    assert total_cost["usd"] == calculated_usd

def test_performance():
    """Test performance metrics"""
    audio_path = os.path.join(TEST_DIR, "test.wav")
    response = client.post(
        "/detect/language",
        json={
            "audio_file_path": audio_path,
            "ground_truth_language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check timing metrics
    for result in data["results"]:
        assert "time_taken" in result
        assert isinstance(result["time_taken"], (int, float))
        assert result["time_taken"] >= 0
        
        # Basic performance thresholds
        if result["provider"] == "whisper_local":
            assert result["time_taken"] < 5.0  # Should complete within 5 seconds
        elif result["provider"] in ["sarvam_mock", "eleven_mock"]:
            assert result["time_taken"] < 1.0  # Mock services should be fast 