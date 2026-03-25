# insideout

Separates audio files into stems (vocals, drums, bass, other), detects BPM and key, and packages everything into a zip or folder. Optionally trims silence from stems.

## Setup

**Requirements:** Python 3.11 (strictly — 3.12+ is not supported due to dependency constraints), [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install uv (if you don't have it)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
# 1. Clone the repo and enter it
git clone <repo-url>
cd insideout

# 2. Create a Python 3.11 virtual environment
uv venv

# 3. Install most dependencies from the lockfile
uv sync
```

Some packages (`llvmlite`, `numba`, `essentia`) require binary-only wheels and can't be built from source on all platforms. If `uv sync` fails or these packages are missing, install them manually:

```bash
.venv/bin/pip install llvmlite --only-binary :all:
.venv/bin/pip install numba --only-binary :all:
.venv/bin/pip install librosa essentia pydub soundfile
```

> **Note for macOS Intel (x86_64):** `llvmlite 0.46+` has no macOS x86_64 wheel. The lockfile pins `llvmlite<0.46` to avoid this — the binary-only flag above ensures pip doesn't attempt to compile it from source.

### Activate the environment

```bash
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

After this, `python main.py` is ready to use.

## Usage

```bash
python main.py <input> [options]
```

`input` can be a single audio file or a folder of audio files.
Supported formats: `.wav` `.mp3` `.flac` `.aiff` `.aif` `.ogg` `.m4a` `.wma`

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `./output` | Output directory |
| `--model` | `-m` | `htdemucs` | Demucs model (see below) |
| `--no-zip` | | off | Output a folder instead of a zip file |
| `--workers` | `-w` | `1` | Parallel workers for batch processing |

## Models

| Model | Stems | Notes |
|-------|-------|-------|
| `htdemucs` | 4 | vocals, drums, bass, other — default |
| `htdemucs_ft` | 4 | fine-tuned, higher quality, ~4x slower |
| `htdemucs_6s` | 6 | adds guitar and piano separation |
| `mdx_extra` | 4 | alternative architecture, strong on vocals |

## Trimming (silence removal)

When `--trim` is enabled, any stem exceeding the silence threshold gets a `_trimmed` version written alongside the original — the original stem is never overwritten.

| Flag | Default | Description |
|------|---------|-------------|
| `--trim` | off | Enable silence trimming |
| `--trim-stems` | `vocals` | Stems to trim. Comma-separated or `all` |
| `--trim-n` | `20` | Minimum % of file that must be silent to trigger trimming |
| `--trim-m` | `-40` | Silence threshold in dBFS |
| `--trim-s` | `1000` | Minimum silence duration in ms to be cut |
| `--trim-t` | `1500` | Gap in ms inserted between kept chunks |

## Examples

```bash
# Single file, all defaults
python main.py track.mp3

# Single file, output to specific folder
python main.py track.mp3 -o ~/Desktop/stems

# Single file, higher quality model, output as folder instead of zip
python main.py track.mp3 --model htdemucs_ft --no-zip

# Batch process a folder with 3 parallel workers
python main.py ~/Music/tracks/ --workers 3

# Batch with trimming enabled on vocals (default)
python main.py ~/Music/tracks/ --workers 2 --trim

# Trim vocals and other stems
python main.py track.mp3 --trim --trim-stems vocals,other

# Trim all stems with custom settings
python main.py track.mp3 --trim --trim-stems all --trim-m -50 --trim-s 500 --trim-t 2000
```

## Output

For a file `track.mp3` detected as A minor at 128 BPM:

```
track_Amin_128bpm.zip
├── track_Amin_128bpm.mp3              original file, renamed
├── track_Amin_128bpm_vocals.wav
├── track_Amin_128bpm_drums.wav
├── track_Amin_128bpm_bass.wav
└── track_Amin_128bpm_other.wav

# with --trim:
└── track_Amin_128bpm_vocals_trimmed.wav   silence removed, chunks joined with gap
```
