# Spoken Language Detection Service

A robust service that detects spoken language in audio files by integrating multiple AI providers. Built with FastAPI, OpenAI Whisper, and fastText.

## Features

- Multiple language detection providers:
  - OpenAI Whisper (local implementation)
  - fastText Language ID
  - Sarvam AI (mock)
  - ElevenLabs (mock)
- Ensemble decision making
- Cost and performance tracking
- Support for Indian languages
- Robust error handling
- Parallel execution for optimal performance

## System Requirements

- Python 3.10 or 3.11
- FFmpeg
- 500MB+ free disk space
- Modern multi-core CPU

## Installation

1. Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg -y
```

2. Set up Python environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download fastText model:
```bash
curl -L -o models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── models/
│   └── lid.176.bin
├── samples/
│   └── test.wav
└── src/
    ├── main.py
    ├── coordinator.py
    ├── utils/
    │   └── audio.py
    └── connectors/
        ├── whisper_connector.py
        ├── fasttext_lid.py
        ├── sarvam_mock.py
        └── eleven_mock.py
```

## Usage

1. Start the server:
```bash
cd src
uvicorn main:app --reload
```

2. Make a request:
```bash
curl -X POST http://localhost:8000/detect/language \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file_path": "/path/to/audio.wav",
    "ground_truth_language": "en"
  }'
```

## API Response Format

```json
{
  "results": [
    {
      "provider": "whisper_local",
      "language": "en",
      "confidence": 0.95,
      "time_taken": 1.234,
      "transcript": "...",
      "status": "success",
      "error": null,
      "cost": {
        "tokens": 100,
        "usd": 0.001
      }
    },
    // ... results from other providers
  ],
  "ensemble": {
    "final_language": "en",
    "scores": {
      "en": 1.9,
      "hi": 0.7
    },
    "total_cost": {
      "tokens": 250,
      "usd": 0.0025
    }
  }
}
```

## Features

1. **Multiple Providers**: Integrates 2 real and 2 mock language detection services
2. **Cost Tracking**: Estimates token usage and cost for each provider
3. **Performance Metrics**: Tracks execution time and confidence scores
4. **Error Handling**: Robust error handling with detailed status reporting
5. **Ensemble Decision**: Weighted voting system for final language detection
6. **Indian Language Support**: Both Whisper and fastText support Indian languages

## Development

- Uses FastAPI for modern, async API development
- Modular design with separate connectors for each provider
- Parallel execution of providers for optimal performance
- Comprehensive error handling and reporting
- Cost and performance tracking built-in

## Testing

Run tests with:
```bash
pytest
```

Sample audio files are provided in the `samples/` directory for testing.

## Performance Optimization

- Providers run in parallel using asyncio
- Local models (Whisper, fastText) avoid API latency
- Efficient audio preprocessing with FFmpeg
- Caches loaded models for better performance

## License

MIT License 