
import numpy as np

from pedalboard import (
    Pedalboard,
    PitchShift,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    LowShelfFilter,
    Distortion,
    Chorus,
    Compressor,
    Reverb,
)

SAMPLE_RATE = 24000

ultron_effects = Pedalboard([

    # Deep mechanical voice.
    HighpassFilter(
        cutoff_frequency_hz=55
    ),

    PitchShift(
        semitones=-3.5
    ),

    # Add physical weight.
    LowShelfFilter(
        cutoff_frequency_hz=180,
        gain_db=5.0,
        q=0.7
    ),

    # Metallic vocal resonance.
    PeakFilter(
        cutoff_frequency_hz=1800,
        gain_db=4.0,
        q=1.2
    ),

    # Hard metallic presence.
    PeakFilter(
        cutoff_frequency_hz=3500,
        gain_db=3.5,
        q=1.0
    ),

    LowpassFilter(
        cutoff_frequency_hz=9500
    ),

    # Aggressive metallic harmonics.
    Distortion(
        drive_db=7.0
    ),

    # Mechanical vocal doubling.
    Chorus(
        rate_hz=0.30,
        depth=0.25,
        centre_delay_ms=7,
        feedback=0.12,
        mix=0.16
    ),

    # Strong, controlled voice.
    Compressor(
        threshold_db=-20,
        ratio=3.0,
        attack_ms=8,
        release_ms=140
    ),

    # Minimal artificial space.
    Reverb(
        room_size=0.18,
        damping=0.55,
        wet_level=0.05,
        dry_level=0.95,
        width=0.8
    )
])


def process_voice(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    mono = audio.reshape(1, -1)

    processed = ultron_effects(
        mono,
        SAMPLE_RATE
    )

    processed = processed.reshape(-1)

    peak = np.max(np.abs(processed))

    if peak > 0.95:
        processed *= 0.95 / peak

    return processed.astype(np.float32)
