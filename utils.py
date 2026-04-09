from pathlib import Path
from threading import Lock

_print_lock = Lock()
_log_file = None

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".wma"}


def set_log_file(path: str):
    global _log_file
    _log_file = open(path, "w", buffering=1)


def get_log_file():
    return _log_file


def tprint(*args, **kwargs):
    """Thread-safe print, also writes to log file if enabled."""
    with _print_lock:
        print(*args, **kwargs)
        if _log_file:
            print(*args, **kwargs, file=_log_file)


def build_label(bpm: float, key: str, scale: str) -> str:
    scale_short = "min" if scale.lower().startswith("min") else "maj"
    return f"{key}{scale_short}_{int(bpm)}bpm"


def collect_files(input_path: str, recursive: bool = False, exclude_dir: str | None = None) -> list[Path]:
    p = Path(input_path)
    exclude = Path(exclude_dir).resolve() if exclude_dir else None
    if p.is_file():
        return [p]
    if p.is_dir():
        iterator = p.rglob("*") if recursive else p.iterdir()
        files = sorted(
            f for f in iterator
            if f.is_file()
            and f.suffix.lower() in AUDIO_EXTENSIONS
            and (exclude is None or not f.resolve().is_relative_to(exclude))
        )
        if not files:
            raise FileNotFoundError(f"No audio files found in '{p}'")
        return files
    raise FileNotFoundError(f"Path not found: '{p}'")
