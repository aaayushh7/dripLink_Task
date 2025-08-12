import asyncio, time

async def sarvam_mock(audio_path: str):
    start = time.time()
    await asyncio.sleep(0.08)
    duration = time.time() - start
    
    # Mock cost estimates
    tokens = 100  # mock token usage
    cost_per_token = 0.00002  # $0.02 per 1000 tokens
    cost_usd = tokens * cost_per_token
    
    return {
        "provider": "sarvam_mock",
        "language": "hi",
        "confidence": 0.7,
        "time_taken": duration,
        "transcript": None,
        "status": "success",
        "error": None,
        "cost": {
            "tokens": tokens,
            "usd": cost_usd
        }
    } 