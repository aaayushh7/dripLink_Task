from fastapi import FastAPI, HTTPException
from utils.audio import preprocess_audio, get_duration_seconds
from coordinator import run_all
import os
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="Spoken Language Detector")

class DetectionRequest(BaseModel):
    audio_file_path: str
    ground_truth_language: Optional[str] = None

@app.post("/detect/language")
async def detect_language(request: DetectionRequest):
    # Validate file exists
    if not os.path.exists(request.audio_file_path):
        raise HTTPException(status_code=400, detail="Audio file not found")
    
    # Preprocess to 16k wav
    try:
        proc_path = preprocess_audio(request.audio_file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")
    
    # Run connectors / ensemble
    try:
        resp = await run_all(proc_path)
        # Add ground truth if provided
        if request.ground_truth_language:
            resp["ground_truth_language"] = request.ground_truth_language
        return resp
    finally:
        # cleanup processed file
        try:
            os.remove(proc_path)
        except Exception:
            pass 