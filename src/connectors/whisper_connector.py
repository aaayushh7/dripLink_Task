import whisper
import asyncio
import time
from typing import Dict, Any

# Load model once at import (choose 'base' for speed, 'small' or 'medium' for accuracy)
# 'base' is often a good balance on CPU.
MODEL_NAME = "base"  # change to 'small' or 'medium' if you have more CPU/RAM
_model = whisper.load_model(MODEL_NAME)

async def transcribe_with_whisper(audio_path: str) -> Dict[str, Any]:
    """
    Runs whisper transcription in a thread (blocking) and returns result dict.
    """
    start = time.time()
    loop = asyncio.get_running_loop()
    def _sync():
        # returns dict with 'text' and 'language' keys among others
        return _model.transcribe(audio_path)
    try:
        resp = await loop.run_in_executor(None, _sync)
        lang = resp.get("language")
        text = resp.get("text", "")
        duration = time.time() - start
        
        # Estimate cost (this is a local model, so only compute cost)
        # Assuming average cloud GPU cost of $0.50 per hour
        cost_usd = (duration / 3600) * 0.50
        tokens = len(text.split())  # rough estimate
        
        return {
            "provider": "whisper_local",
            "language": lang,
            "confidence": None,
            "time_taken": duration,
            "transcript": text,
            "status": "success",
            "error": None,
            "cost": {
                "tokens": tokens,
                "usd": cost_usd
            }
        }
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