import sys
from pathlib import Path


def separate_stems(audio_path: str, output_dir: str, model: str = "htdemucs") -> Path:
    """Run demucs. Returns the path to the folder containing stem files."""
    import subprocess
    cmd = [
        sys.executable, "-m", "demucs",
        "--out", output_dir,
        "--name", model,
        audio_path,
    ]
    result = subprocess.run(cmd)  # stdout/stderr pass through so tqdm progress bar is visible
    if result.returncode != 0:
        raise RuntimeError("demucs failed — see output above")

    track_name = Path(audio_path).stem
    stems_dir = Path(output_dir) / model / track_name
    if not stems_dir.exists():
        raise RuntimeError(f"Expected stems at {stems_dir} but not found")
    return stems_dir
