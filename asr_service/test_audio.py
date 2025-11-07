#!/usr/bin/env python3
"""
Generate test audio files for ASR testing

Creates synthetic audio files with speech-like characteristics
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_wave(frequency, duration, sample_rate=16000):
    """Generate a sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t)
    return wave


def generate_speech_like_audio(duration=5.0, sample_rate=16000):
    """
    Generate speech-like audio with multiple frequency components
    This simulates the harmonic structure of human speech
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)

    # Fundamental frequency (varies over time to simulate prosody)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # 120-180 Hz

    # Add harmonics (speech has multiple harmonics)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic  # Higher harmonics are quieter
        audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

    # Add formants (resonances typical in speech)
    formants = [800, 1200, 2500]  # Typical vowel formants in Hz
    for formant in formants:
        audio += 0.3 * np.sin(2 * np.pi * formant * t)

    # Add envelope (amplitude modulation to simulate syllables)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # ~3 syllables/sec
    audio = audio * envelope

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, len(audio))
    audio += noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)


def generate_silence(duration=1.0, sample_rate=16000):
    """Generate silence"""
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


def generate_test_file(output_path, duration=5.0, file_type="speech"):
    """Generate a test audio file"""
    sample_rate = 16000

    if file_type == "speech":
        audio = generate_speech_like_audio(duration, sample_rate)
    elif file_type == "sine":
        audio = generate_sine_wave(440, duration, sample_rate)
    elif file_type == "silence":
        audio = generate_silence(duration, sample_rate)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    # Save file
    sf.write(output_path, audio, sample_rate)
    print(f"✅ Generated: {output_path}")
    print(f"   Duration: {duration}s")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument(
        "--output",
        "-o",
        default="test_audio.wav",
        help="Output file path (default: test_audio.wav)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["speech", "sine", "silence"],
        default="speech",
        help="Type of audio to generate (default: speech)",
    )

    args = parser.parse_args()

    generate_test_file(args.output, args.duration, args.type)


if __name__ == "__main__":
    main()
