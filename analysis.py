def load_audio(audio_path: str, sample_rate: int = 44100):
    """Load mono audio via essentia."""
    import essentia.standard as es
    return es.MonoLoader(filename=audio_path, sampleRate=sample_rate)()


def detect_bpm(audio) -> float:
    """Use essentia RhythmExtractor2013, then correct octave errors.

    The extractor sometimes returns 2x or 0.5x the true tempo.
    We fold into a target window of 70–140 BPM which covers most music.
    Adjust BPM_MIN / BPM_MAX if you work primarily with faster genres
    (e.g. drum & bass, techno) — set BPM_MAX to 175 in that case.
    """
    import essentia.standard as es
    BPM_MIN, BPM_MAX = 70, 140

    extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, _, _, _, _ = extractor(audio)
    bpm = float(bpm)

    while bpm > BPM_MAX:
        bpm /= 2
    while bpm < BPM_MIN:
        bpm *= 2

    return round(bpm, 1)


def detect_key(stem_path: str) -> tuple[str, str]:
    """Detect key using the Krumhansl-Schmuckler algorithm on a harmonic stem.

    Steps:
      1. Load the stem with librosa and extract only the harmonic component (HPSS)
         to strip any remaining percussion transients.
      2. Compute chroma_cqt at 36 bins/octave — more pitch-accurate than STFT chroma.
      3. Correlate the mean chroma vector against all 24 K-S major/minor profiles
         and return the key with the highest Pearson correlation.
    """
    import librosa
    import numpy as np

    # Krumhansl-Schmuckler tonal hierarchy profiles
    MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    KEYS  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    y, sr = librosa.load(stem_path, mono=True)

    # Strip remaining transients — harmonic component only
    y_harm = librosa.effects.harmonic(y, margin=4)

    # CQT chroma at 3× resolution, then reduce to 12 bins for KS
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=36)
    mean_chroma = chroma.mean(axis=1)  # shape (12,)

    best_key, best_scale, best_corr = "C", "major", -np.inf
    for i, key in enumerate(KEYS):
        for profile, scale in [(np.roll(MAJOR, i), "major"), (np.roll(MINOR, i), "minor")]:
            corr = np.corrcoef(mean_chroma, profile)[0, 1]
            if corr > best_corr:
                best_corr, best_key, best_scale = corr, key, scale

    return best_key, best_scale
