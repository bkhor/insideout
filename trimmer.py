from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrimConfig:
    stems: list[str]          # which stem names to apply to, e.g. ["vocals"] or ["vocals", "other"]
    silence_pct: float = 20.0 # N — only trim if silence makes up at least this % of the file
    db_threshold: float = -40 # M — dB level below which audio is considered silent
    min_silence_ms: int = 1000 # S — silence must last at least this long (ms) to be cut
    gap_ms: int = 1500         # T — gap inserted between kept chunks (ms)
    min_chunk_ms: int = 500    # chunks shorter than this are discarded (not user-facing)


def should_apply(stem_name: str, config: TrimConfig) -> bool:
    """Check whether trimming should be applied to a given stem name."""
    name = stem_name.lower()
    return any(s == "all" or s in name for s in config.stems)


def trim_stem(stem_path: str, out_path: str, config: TrimConfig) -> bool:
    """Trim silent regions from a stem file and write the result to out_path.

    Returns True if trimming was applied (silence% exceeded threshold),
    False if the file was quiet enough that trimming wasn't needed.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_file(stem_path)

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=config.min_silence_ms,
        silence_thresh=config.db_threshold,
        seek_step=10,
    )

    if not nonsilent_ranges:
        return False

    total_ms = len(audio)
    sound_ms = sum(end - start for start, end in nonsilent_ranges)
    silence_pct = (1 - sound_ms / total_ms) * 100

    if silence_pct < config.silence_pct:
        return False

    # Filter out chunks too short to be meaningful
    chunks = [
        audio[start:end]
        for start, end in nonsilent_ranges
        if (end - start) >= config.min_chunk_ms
    ]

    if not chunks:
        return False

    gap = AudioSegment.silent(duration=config.gap_ms)
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result + gap + chunk

    suffix = Path(out_path).suffix.lstrip(".")
    fmt = "wav" if suffix in ("wav", "") else suffix
    result.export(out_path, format=fmt)
    return True


def parse_trim_stems(value: str) -> list[str]:
    """Parse --trim-stems CLI value into a list of stem names."""
    return [s.strip().lower() for s in value.split(",")]
