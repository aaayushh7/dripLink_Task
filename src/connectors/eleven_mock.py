import asyncio, time

async def eleven_mock(audio_path: str):
    start = time.time()
    await asyncio.sleep(0.1)
    duration = time.time() - start
    
    # Mock cost estimates
    tokens = 150  # mock token usage
    cost_per_token = 0.00003  # $0.03 per 1000 tokens
    cost_usd = tokens * cost_per_token
    
    return {
        "provider": "eleven_mock",
        "language": "en",
        "confidence": 0.9,
        "time_taken": duration,
        "transcript": None,
        "status": "success",
        "error": None,
        "cost": {
            "tokens": tokens,
            "usd": cost_usd
        }
    } 