from pathlib import Path
from threading import Lock

_print_lock = Lock()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".wma"}


def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def build_label(bpm: float, key: str, scale: str) -> str:
    scale_short = "min" if scale.lower().startswith("min") else "maj"
    return f"{key}{scale_short}_{int(bpm)}bpm"


def collect_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No audio files found in '{p}'")
        return files
    raise FileNotFoundError(f"Path not found: '{p}'")
