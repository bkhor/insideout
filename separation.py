import subprocess
import sys
from pathlib import Path

from utils import get_log_file


def separate_stems(audio_path: str, output_dir: str, model: str = "htdemucs") -> Path:
    """Run demucs. Returns the path to the folder containing stem files."""
    cmd = [
        sys.executable, "-m", "demucs",
        "--out", output_dir,
        "--name", model,
        audio_path,
    ]
    log_file = get_log_file()
    if log_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        proc.wait()
        returncode = proc.returncode
    else:
        returncode = subprocess.run(cmd).returncode  # stdout/stderr pass through so tqdm progress bar is visible
    if returncode != 0:
        raise RuntimeError("demucs failed — see output above")

    track_name = Path(audio_path).stem
    stems_dir = Path(output_dir) / model / track_name
    if not stems_dir.exists():
        raise RuntimeError(f"Expected stems at {stems_dir} but not found")
    return stems_dir
