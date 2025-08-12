import asyncio
from typing import List, Dict, Any
from connectors.whisper_connector import transcribe_with_whisper
from connectors.fasttext_lid import detect_language_fasttext
from connectors.sarvam_mock import sarvam_mock
from connectors.eleven_mock import eleven_mock

CONNECTORS = [
    transcribe_with_whisper,
    sarvam_mock,
    eleven_mock
]

async def run_all(audio_path: str) -> Dict[str, Any]:
    """
    Runs all language detection providers and returns aggregated results.
    Also runs fastText on any available transcripts for additional language detection.
    Gives higher weight to audio-based detectors over text-based ones.
    """
    # Run real and mock connectors in parallel
    tasks = [asyncio.create_task(c(audio_path)) for c in CONNECTORS]
    done = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    whisper_transcript = None
    
    # Process results and handle exceptions
    for r in done:
        if isinstance(r, Exception):
            results.append({
                "provider": "unknown",
                "language": None,
                "confidence": None,
                "time_taken": None,
                "transcript": None,
                "status": "error",
                "error": str(r),
                "cost": {"tokens": 0, "usd": 0}
            })
        else:
            results.append(r)
            if r.get("provider") == "whisper_local" and r.get("transcript"):
                whisper_transcript = r.get("transcript")

    # Run fastText on any available transcript
    if whisper_transcript:
        fasttext_result = await detect_language_fasttext(whisper_transcript)
        results.append(fasttext_result)

    # Ensemble: weighted vote by confidence and provider type
    votes = {}
    for r in results:
        if r["status"] != "success":
            continue
        lang = r.get("language")
        if not lang:
            continue
            
        # Base weight is confidence or 1.0
        w = r.get("confidence") or 1.0
        
        # Give higher weight to audio-based detectors
        if r["provider"] in ["whisper_local", "sarvam_mock"]:
            w *= 2.0  # Double weight for audio-based detectors
            
        votes[lang] = votes.get(lang, 0.0) + w
    
    # Calculate total costs
    total_cost = {
        "tokens": sum(r.get("cost", {}).get("tokens", 0) for r in results),
        "usd": sum(r.get("cost", {}).get("usd", 0) for r in results)
    }
    
    # Find the winning language
    final = None
    if votes:
        final = max(votes.items(), key=lambda kv: kv[1])[0]
    
    return {
        "results": results,
        "ensemble": {
            "final_language": final,
            "scores": votes,
            "total_cost": total_cost
        }
    } 