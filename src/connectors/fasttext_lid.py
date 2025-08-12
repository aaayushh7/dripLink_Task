import fasttext
import os, time
from typing import Tuple, Dict, Any
from pathlib import Path

# Get absolute path to the model file
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = os.environ.get("FASTTEXT_LID_MODEL", str(PROJECT_ROOT / "models" / "lid.176.bin"))
_lid = None

def load_model():
    global _lid
    if _lid is None:
        _lid = fasttext.load_model(MODEL_PATH)
    return _lid

async def detect_language_fasttext(text: str) -> Dict[str, Any]:
    """
    Detects language using fastText and returns a standardized response.
    """
    start = time.time()
    try:
        model = load_model()
        labels, probs = model.predict(text, k=1)
        label = labels[0].replace("__label__", "")
        prob = float(probs[0])
        duration = time.time() - start
        
        # Estimate cost (this is a local model, so minimal cost)
        tokens = len(text.split())
        cost_per_token = 0.000001  # $0.001 per 1000 tokens
        cost_usd = tokens * cost_per_token
        
        return {
            "provider": "fasttext_local",
            "language": label,
            "confidence": prob,
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
            "provider": "fasttext_local",
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