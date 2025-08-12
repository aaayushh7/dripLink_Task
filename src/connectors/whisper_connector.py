import whisper
import asyncio
import time
from typing import Dict, Any

# Load model once at import (using base model for balance of speed and accuracy)
MODEL_NAME = "base"  # Smaller size, still good for Indian language support
_model = whisper.load_model(MODEL_NAME)

# Define Indian language codes and their scripts
INDIAN_LANGUAGES = {
    "hi": "Devanagari",  # Hindi
    "bn": "Bengali",     # Bengali
    "ta": "Tamil",       # Tamil
    "te": "Telugu",      # Telugu
    "mr": "Devanagari",  # Marathi
    "gu": "Gujarati",    # Gujarati
    "kn": "Kannada",     # Kannada
    "ml": "Malayalam",   # Malayalam
    "pa": "Gurmukhi",    # Punjabi
    "or": "Oriya",       # Odia
    "as": "Bengali",     # Assamese
    "ur": "Urdu"         # Urdu
}

async def transcribe_with_whisper(audio_path: str) -> Dict[str, Any]:
    """
    Runs whisper transcription in a thread (blocking) and returns result dict.
    Optimized for Indian language detection and native script preservation.
    """
    start = time.time()
    loop = asyncio.get_running_loop()
    def _sync():
        # First detect language
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(_model.device)
        _, probs = _model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        # Configure transcription options
        decode_options = {
            "without_timestamps": True,
            "fp16": False
        }
        
        # Special handling for Indian languages
        if detected_lang in INDIAN_LANGUAGES:
            decode_options.update({
                "suppress_tokens": [1],  # Suppress English token
                "language": detected_lang,  # Force detected language
            })
        
        result = _model.transcribe(
            audio_path,
            task="transcribe",
            **decode_options
        )
        result["language_confidence"] = probs[detected_lang]
        
        # Add script information for Indian languages
        if detected_lang in INDIAN_LANGUAGES:
            result["script"] = INDIAN_LANGUAGES[detected_lang]
        
        return result

    try:
        resp = await loop.run_in_executor(None, _sync)
        lang = resp.get("language")
        text = resp.get("text", "")
        duration = time.time() - start
        
        # Estimate cost (this is a local model, so only compute cost)
        # Assuming average cloud GPU cost of $0.50 per hour
        cost_usd = (duration / 3600) * 0.50
        tokens = len(text.split())  # rough estimate
        
        response = {
            "provider": "whisper_local",
            "language": lang,
            "confidence": resp.get("language_confidence"),
            "time_taken": duration,
            "transcript": text,
            "status": "success",
            "error": None,
            "cost": {
                "tokens": tokens,
                "usd": cost_usd
            }
        }
        
        # Add script information if available
        if "script" in resp:
            response["script"] = resp["script"]
        
        return response
    except Exception as e:
        return {
            "provider": "whisper_local",
            "language": None,
            "confidence": None,
            "time_taken": time.time() - start,
            "transcript": None,
            "status": "error",
            "error": str(e),
            "cost": {
                "tokens": 0,
                "usd": 0
            }
        } 