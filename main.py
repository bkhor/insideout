"""
insideout — audio stem separator with BPM and key detection

Usage:
    python main.py track.mp3
    python main.py track.wav --output ./out --no-zip
    python main.py ~/Music/tracks/ --workers 3 --model htdemucs_ft
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from analysis import detect_bpm, detect_key, load_audio
from separation import separate_stems
from trimmer import TrimConfig, parse_trim_stems, should_apply, trim_stem
from utils import AUDIO_EXTENSIONS, build_label, collect_files, tprint

FORMATS = ("mp3", "wav", "flac")


def write_audio(src: Path, dest_no_ext: Path, fmt: str) -> Path:
    """Copy/convert a WAV stem file to the target format. Returns the written path."""
    dest = dest_no_ext.with_suffix(f".{fmt}")
    if fmt == "wav":
        shutil.copy2(src, dest)
    elif fmt == "flac":
        data, rate = sf.read(str(src))
        sf.write(str(dest), data, rate, format="FLAC")
    elif fmt == "mp3":
        seg = AudioSegment.from_wav(str(src))
        seg.export(str(dest), format="mp3")
    return dest


def write_array(data: np.ndarray, rate: int, dest_no_ext: Path, fmt: str) -> Path:
    """Write a numpy audio array to the target format. Returns the written path."""
    dest = dest_no_ext.with_suffix(f".{fmt}")
    if fmt == "mp3":
        # soundfile can't write MP3; write a temp WAV then convert
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, data, rate)
            seg = AudioSegment.from_wav(tmp_path)
            seg.export(str(dest), format="mp3")
        finally:
            os.unlink(tmp_path)
    else:
        sf_fmt = "FLAC" if fmt == "flac" else "WAV"
        sf.write(str(dest), data, rate, format=sf_fmt)
    return dest


def process(audio_path: str, output_dir: str, model: str, make_zip: bool, fmt: str = "wav", trim_config: TrimConfig | None = None):
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    name = Path(audio_path).name
    stem_name = Path(audio_path).stem
    suffix = Path(audio_path).suffix
    tag = f"[{name}]"

    # --- Skip if already processed ---
    existing = list(Path(output_dir).glob(f"{stem_name}_*"))
    if existing:
        tprint(f"{tag} Already processed (found {existing[0].name}) — skipping")
        return

    # --- Step 1: BPM from full mix ---
    tprint(f"{tag} Detecting BPM...")
    full_audio = load_audio(audio_path)
    bpm = detect_bpm(full_audio)
    tprint(f"{tag} BPM: {bpm}")

    # --- Step 2: Stem separation ---
    tprint(f"{tag} Separating stems...")
    with tempfile.TemporaryDirectory() as tmp:
        stems_dir = separate_stems(audio_path, tmp, model)
        stem_files = sorted(stems_dir.glob("*"))
        tprint(f"{tag} Stems: {[f.name for f in stem_files]}")

        # --- Step 3: Key from harmonic stem ---
        # Prefer: other > vocals > bass (anything but drums)
        key_stem = None
        for preferred in ("other", "vocals", "bass"):
            matches = [f for f in stem_files if preferred in f.stem.lower()]
            if matches:
                key_stem = matches[0]
                break
        if key_stem is None:
            key_stem = stem_files[0]

        tprint(f"{tag} Detecting key from '{key_stem.name}'...")
        key, scale = detect_key(str(key_stem))
        tprint(f"{tag} Key: {key} {scale}")

        label = build_label(bpm, key, scale)
        out_name = f"{stem_name}_{label}"
        tprint(f"{tag} Label: {label}")

        # --- Package ---
        final_dir = Path(output_dir) / out_name
        final_dir.mkdir(parents=True, exist_ok=True)

        # --- Build instrumental (all stems except vocals) ---
        instrumental_stems = [f for f in stem_files if "vocals" not in f.stem.lower()]
        if instrumental_stems:
            arrays, rate = [], None
            for stem_file in instrumental_stems:
                data, rate = sf.read(str(stem_file))
                arrays.append(data)
            instrumental = np.clip(sum(arrays), -1.0, 1.0)
            write_array(instrumental, rate, final_dir / f"{out_name}_instrumental", fmt)
            tprint(f"{tag} Instrumental written ({len(instrumental_stems)} stems mixed)")

        for stem_file in stem_files:
            write_audio(stem_file, final_dir / f"{out_name}_{stem_file.stem}", fmt)

            # Trimming — produce _trimmed file alongside original stem
            if trim_config and should_apply(stem_file.stem, trim_config):
                trimmed_wav = final_dir / f"_tmp_{stem_file.stem}_trimmed.wav"
                applied = trim_stem(str(stem_file), str(trimmed_wav), trim_config)
                if applied:
                    trimmed_dest = write_audio(trimmed_wav, final_dir / f"{out_name}_{stem_file.stem}_trimmed", fmt)
                    trimmed_wav.unlink(missing_ok=True)
                    tprint(f"{tag} Trimmed '{stem_file.stem}' -> {trimmed_dest.name}")
                else:
                    trimmed_wav.unlink(missing_ok=True)
                    tprint(f"{tag} '{stem_file.stem}' has less than {trim_config.silence_pct}% silence — skipped trimming")

    if make_zip:
        zip_path = Path(output_dir) / f"{out_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(final_dir.iterdir()):
                zf.write(f, f.name)
        shutil.rmtree(final_dir)
        tprint(f"{tag} Done -> {zip_path}")
    else:
        tprint(f"{tag} Done -> {final_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Separate audio into stems and tag filename with key + BPM"
    )
    parser.add_argument("input", help="Input audio file or folder of audio files")
    parser.add_argument(
        "--output", "-o",
        default="./stems",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--model", "-m",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
        help="Demucs model. htdemucs_ft = fine-tuned/higher quality. htdemucs_6s = 6 stems incl. guitar+piano.",
    )
    parser.add_argument(
        "--format", "-f",
        default="wav",
        choices=FORMATS,
        help="Output audio format for stems (default: wav).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Output a folder instead of a zip file",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively scan subdirectories for audio files.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers for batch processing (default: 1). "
             "2-4 is practical on most CPUs; more helps if you have many cores.",
    )

    # Trimming options
    trim = parser.add_argument_group("trimming (silence removal)")
    trim.add_argument(
        "--trim",
        action="store_true",
        help="Enable silence trimming on stems.",
    )
    trim.add_argument(
        "--trim-stems",
        default="vocals",
        metavar="STEMS",
        help="Comma-separated stems to trim, or 'all' (default: vocals). "
             "Examples: 'vocals', 'vocals,other', 'all'.",
    )
    trim.add_argument(
        "--trim-n",
        type=float,
        default=20.0,
        metavar="N",
        help="Minimum %% of file that must be silent to trigger trimming (default: 20).",
    )
    trim.add_argument(
        "--trim-m",
        type=float,
        default=-40.0,
        metavar="M",
        help="Silence threshold in dBFS — quieter than this is considered silent (default: -40).",
    )
    trim.add_argument(
        "--trim-s",
        type=int,
        default=1000,
        metavar="S",
        help="Minimum silence duration in ms to be cut (default: 1000).",
    )
    trim.add_argument(
        "--trim-t",
        type=int,
        default=1500,
        metavar="T",
        help="Gap in ms inserted between kept chunks (default: 1500).",
    )
    args = parser.parse_args()

    trim_config = None
    if args.trim:
        trim_config = TrimConfig(
            stems=parse_trim_stems(args.trim_stems),
            silence_pct=args.trim_n,
            db_threshold=args.trim_m,
            min_silence_ms=args.trim_s,
            gap_ms=args.trim_t,
        )

    files = collect_files(args.input, recursive=args.recursive, exclude_dir=args.output)
    total = len(files)
    workers = min(args.workers, total)

    print(f"Found {total} file(s). Processing with {workers} worker(s).")

    failed = 0
    if workers == 1:
        for i, audio_file in enumerate(files, 1):
            print(f"\n--- [{i}/{total}] {audio_file.name} ---")
            try:
                process(str(audio_file), args.output, args.model, not args.no_zip, args.format, trim_config)
            except Exception as e:
                print(f"ERROR: {e} — skipping")
                failed += 1
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for audio_file in files:
                future = pool.submit(process, str(audio_file), args.output, args.model, not args.no_zip, args.format, trim_config)
                futures[future] = audio_file.name

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                tprint(f"ERROR [{name}]: {e}")
                failed += 1

    status = f"{total - failed}/{total} succeeded"
    if failed:
        status += f", {failed} failed"
    print(f"\nAll done. {status} -> {args.output}/")


if __name__ == "__main__":
    main()
