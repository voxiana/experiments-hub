#!/usr/bin/env python3
"""
Bootstrap Voice Samples for XTTS v2

Downloads and prepares reference voice samples for voice cloning.
This script downloads high-quality speech samples from public datasets
to use as reference voices for XTTS v2.

IMPORTANT: XTTS v2 requires REAL speech samples - synthetic audio will not work.
"""

import os
import sys
import logging
import urllib.request
import tempfile
import subprocess

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

# Voice configurations with sample URLs from Mozilla Common Voice or similar
# These are placeholder URLs - replace with actual public domain samples
VOICE_CONFIGS = {
    "arabic_gulf_male": {
        "language": "ar",
        "description": "Arabic (Gulf) male voice",
    },
    "arabic_gulf_female": {
        "language": "ar",
        "description": "Arabic (Gulf) female voice",
    },
    "arabic_msa_male": {
        "language": "ar",
        "description": "Arabic (MSA) male voice",
    },
    "english_uae_male": {
        "language": "en",
        "description": "English (UAE) male voice",
    },
    "english_uae_female": {
        "language": "en",
        "description": "English (UAE) female voice",
    },
}


def convert_to_required_format(input_path: str, output_path: str) -> bool:
    """
    Convert audio to the required format for XTTS v2.
    
    Requirements:
    - WAV format
    - 24kHz sample rate
    - Mono channel
    - 6-12 seconds duration
    """
    try:
        # Use ffmpeg for conversion (more robust)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to soundfile
        try:
            audio, sr = sf.read(input_path)
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed (simple approach)
            if sr != SAMPLE_RATE:
                logger.warning(f"Sample rate is {sr}, resampling to {SAMPLE_RATE}")
                # Simple resampling - for better quality, use librosa
                ratio = SAMPLE_RATE / sr
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
            
            sf.write(output_path, audio.astype(np.float32), SAMPLE_RATE)
            return True
        except Exception as e:
            logger.error(f"Failed to convert audio: {e}")
            return False


def validate_audio_file(path: str) -> tuple[bool, str]:
    """
    Validate that an audio file meets XTTS v2 requirements.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not os.path.exists(path):
        return False, "File not found"
    
    try:
        audio, sr = sf.read(path)
        
        # Check sample rate
        if sr != SAMPLE_RATE:
            return False, f"Wrong sample rate: {sr} (expected {SAMPLE_RATE})"
        
        # Check channels
        if len(audio.shape) > 1:
            return False, f"Not mono: {audio.shape[1]} channels"
        
        # Check duration
        duration = len(audio) / sr
        if duration < 3:
            return False, f"Too short: {duration:.1f}s (minimum 3s)"
        if duration > 30:
            return False, f"Too long: {duration:.1f}s (maximum 30s)"
        
        # Check amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp < 0.01:
            return False, f"Audio appears silent (max amplitude: {max_amp:.4f})"
        
        # Check for actual content (not just noise)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            return False, f"Audio RMS too low: {rms:.4f}"
        
        return True, f"Valid ({duration:.1f}s, max_amp={max_amp:.3f})"
        
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_existing_voices() -> dict[str, bool]:
    """Check which voice files already exist and are valid"""
    os.makedirs(VOICES_DIR, exist_ok=True)
    
    results = {}
    for voice_id in list(VOICE_CONFIGS.keys()) + ["reference_voice"]:
        path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        is_valid, msg = validate_audio_file(path)
        results[voice_id] = is_valid
        if is_valid:
            logger.info(f"  ✅ {voice_id}: {msg}")
        else:
            logger.info(f"  ❌ {voice_id}: {msg}")
    
    return results


def create_placeholder_voice(voice_id: str, config: dict) -> bool:
    """
    Create a placeholder voice file that explains the requirement.
    
    NOTE: This creates a silent file as a placeholder.
    The TTS will produce noise with this - user must replace with real samples.
    """
    output_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    # Create 6 seconds of near-silence (not completely silent to avoid errors)
    duration = 6.0
    samples = int(duration * SAMPLE_RATE)
    
    # Very low amplitude noise (placeholder)
    audio = np.random.randn(samples).astype(np.float32) * 0.001
    
    sf.write(output_path, audio, SAMPLE_RATE)
    logger.warning(
        f"  ⚠️  Created placeholder for {voice_id}. "
        f"Replace with real speech sample!"
    )
    return True


def copy_reference_to_missing(reference_path: str, missing_voices: list[str]) -> int:
    """
    Copy a valid reference voice to fill in missing voices.
    This allows the service to start, but all voices will sound the same.
    """
    if not os.path.exists(reference_path):
        return 0
    
    count = 0
    for voice_id in missing_voices:
        output_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        try:
            # Read and write (don't just copy, to ensure format is correct)
            audio, sr = sf.read(reference_path)
            sf.write(output_path, audio, sr)
            logger.info(f"  📋 Copied reference to {voice_id}")
            count += 1
        except Exception as e:
            logger.error(f"  ❌ Failed to copy to {voice_id}: {e}")
    
    return count


def main():
    """Main bootstrap function"""
    logger.info("=" * 60)
    logger.info("XTTS v2 Voice Sample Bootstrap")
    logger.info("=" * 60)
    logger.info("")
    
    # Check existing voices
    logger.info("Checking existing voice files...")
    existing = check_existing_voices()
    
    valid_count = sum(existing.values())
    total_count = len(existing)
    
    logger.info("")
    logger.info(f"Found {valid_count}/{total_count} valid voice files")
    
    if valid_count == total_count:
        logger.info("")
        logger.info("✅ All voice files are present and valid!")
        return 0
    
    # Find missing voices
    missing = [v for v, valid in existing.items() if not valid]
    
    logger.info("")
    logger.info(f"Missing or invalid voices: {', '.join(missing)}")
    logger.info("")
    
    # Strategy 1: If reference_voice exists, copy it to missing voices
    reference_path = os.path.join(VOICES_DIR, "reference_voice.wav")
    if existing.get("reference_voice", False):
        logger.info("Using reference_voice.wav as template for missing voices...")
        non_ref_missing = [v for v in missing if v != "reference_voice"]
        copied = copy_reference_to_missing(reference_path, non_ref_missing)
        if copied > 0:
            logger.info(f"  Copied reference to {copied} voice(s)")
            logger.warning(
                "\n⚠️  WARNING: All copied voices will sound identical.\n"
                "   Replace with unique voice samples for variety.\n"
            )
    else:
        # Create placeholders
        logger.warning("No valid reference_voice.wav found.")
        logger.warning("Creating placeholder files (TTS will produce noise!)...")
        logger.warning("")
        
        for voice_id in missing:
            if voice_id in VOICE_CONFIGS:
                create_placeholder_voice(voice_id, VOICE_CONFIGS[voice_id])
            elif voice_id == "reference_voice":
                create_placeholder_voice(voice_id, {"language": "ar"})
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Bootstrap Complete")
    logger.info("=" * 60)
    logger.info("")
    
    # Re-check
    final_check = check_existing_voices()
    final_valid = sum(final_check.values())
    
    if final_valid < total_count:
        logger.error("")
        logger.error("⚠️  IMPORTANT: Voice files are placeholders or copies!")
        logger.error("")
        logger.error("For proper TTS output, add real voice samples to:")
        logger.error(f"  {VOICES_DIR}/")
        logger.error("")
        logger.error("Requirements for each voice file:")
        logger.error("  - Format: WAV")
        logger.error("  - Sample rate: 24,000 Hz")
        logger.error("  - Channels: Mono (1)")
        logger.error("  - Duration: 6-12 seconds")
        logger.error("  - Content: Clear speech in target language")
        logger.error("")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
