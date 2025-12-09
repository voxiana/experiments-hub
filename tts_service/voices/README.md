# Voice Samples Directory

This directory contains reference audio files for XTTS v2 voice cloning.

## ⚠️ IMPORTANT

**XTTS v2 requires REAL speech samples** - synthetic audio will produce silence or noise (the "zzzzz" sound).

## Required Files

Add the following voice sample files to this directory:

- `arabic_gulf_male.wav` - Arabic (Gulf) male voice
- `arabic_gulf_female.wav` - Arabic (Gulf) female voice  
- `arabic_msa_male.wav` - Arabic (MSA) male voice
- `english_uae_male.wav` - English (UAE) male voice
- `english_uae_female.wav` - English (UAE) female voice

## File Requirements

Each voice sample must meet these specifications:

- **Format**: WAV (uncompressed)
- **Sample Rate**: 24,000 Hz (24 kHz)
- **Channels**: Mono (1 channel)
- **Duration**: 6-12 seconds
- **Content**: Clear, natural speech in the target language
- **Quality**: High quality, minimal background noise

## How to Create Voice Samples

### Option 1: Bootstrap Using TTS Model (Quick Start)

**⚠️ This generates initial samples using the TTS model itself. Quality may vary. For production, use Option 2 or 3.**

Run the bootstrap script to generate initial voice samples:

```bash
# Inside the Docker container or local environment
cd tts_service
python3 bootstrap_voices.py
```

This will:
1. Load the XTTS v2 model
2. Generate bootstrap reference audio
3. Create voice samples for all 5 voices
4. Save them to the `voices/` directory

**Note**: These are bootstrap samples. Replace with real recordings for production use.

### Option 2: Record Your Own

1. Use a good microphone in a quiet environment
2. Record 6-12 seconds of clear speech
3. Convert to WAV format, 24kHz, mono:
   ```bash
   # Using ffmpeg
   ffmpeg -i input.mp3 -ar 24000 -ac 1 -f wav output.wav
   ```

### Option 3: Use Existing Audio

1. Find or extract 6-12 seconds of clear speech
2. Convert to the required format:
   ```bash
   ffmpeg -i input.wav -ar 24000 -ac 1 -f wav output.wav
   ```

### Option 4: Download Sample Voices

You can find sample voices from:
- Coqui TTS community samples
- Public domain speech datasets
- Your own recordings

## Verification

After adding voice samples, restart the TTS service. The startup logs will show:

```
✅ arabic_gulf_male: voices/arabic_gulf_male.wav (found, 8.5s)
✅ arabic_gulf_female: voices/arabic_gulf_female.wav (found, 7.2s)
...
✅ All 5 voice preset files found and valid
```

If files are missing or invalid, you'll see warnings with instructions.

## Troubleshooting

### "zzzzz" sound / silence output

This means the reference audio is invalid (synthetic or corrupted). Solution:
1. Delete any `default_*_reference.wav` files
2. Add proper voice samples as described above
3. Restart the service

### "Reference audio file not found"

Add the missing voice sample file to this directory with the exact name shown in the error.

### Poor voice quality

- Ensure the sample is 6-12 seconds (not too short or too long)
- Use high-quality recordings with minimal noise
- Ensure correct format (24kHz, mono, WAV)

## Notes

- Files in this directory are **not** tracked in git (see `.gitignore`)
- Default reference files (`default_*_reference.wav`) are auto-generated fallbacks and will produce poor quality
- For production, always use real voice samples

