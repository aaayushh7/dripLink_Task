import subprocess
from pathlib import Path
import shlex

def preprocess_audio(input_path: str, output_path: str | None = None) -> str:
    """
    Convert any audio to 16kHz mono WAV using ffmpeg CLI.
    Returns the path to the processed file.
    """
    p = Path(input_path)
    if output_path is None:
        output_path = str(p.with_suffix(".proc.wav"))
    cmd = f"ffmpeg -y -i {shlex.quote(str(input_path))} -ac 1 -ar 16000 -vn {shlex.quote(output_path)}"
    subprocess.run(cmd, shell=True, check=True)
    return output_path

def get_duration_seconds(wav_path: str) -> float:
    # Use soundfile (installed by requirements) or ffprobe; here minimal: librosa is heavier
    import soundfile as sf
    info = sf.info(wav_path)
    return float(info.frames) / info.samplerate 