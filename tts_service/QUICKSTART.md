# TTS Service Quick Start

## The Problem: "zzzzz" Sound / Silence Output

If you're hearing silence or noise ("zzzzz") from the TTS service, it's because **voice sample files are missing**. XTTS v2 requires real speech samples to work.

## Quick Fix: Bootstrap Voice Samples

### Option 1: Run Bootstrap Script (Recommended for Testing)

```bash
# If running locally
cd tts_service
python3 bootstrap_voices.py

# If running in Docker
docker-compose exec tts-service python3 /app/bootstrap_voices.py

# Or if using docker run
docker exec -it voiceai-tts python3 /app/bootstrap_voices.py
```

This will generate initial voice samples using the TTS model itself.

### Option 2: Add Your Own Voice Samples

1. Create voice sample files (WAV, 24kHz, mono, 6-12 seconds)
2. Place them in `tts_service/voices/`:
   - `arabic_gulf_male.wav`
   - `arabic_gulf_female.wav`
   - `arabic_msa_male.wav`
   - `english_uae_male.wav`
   - `english_uae_female.wav`

3. Restart the service

## Verification

After adding voice samples, check the startup logs:

```
✅ arabic_gulf_male: voices/arabic_gulf_male.wav (found, 8.5s)
✅ arabic_gulf_female: voices/arabic_gulf_female.wav (found, 7.2s)
...
✅ All 5 voice preset files found and valid
```

If you see ❌ warnings, the files are missing or invalid.

## Docker Setup

If using Docker Compose, you can bootstrap voices after the container starts:

```bash
# Start the service
docker-compose up -d tts-service

# Wait for model to load, then bootstrap voices
docker-compose exec tts-service python3 /app/bootstrap_voices.py

# Restart to verify
docker-compose restart tts-service
```

## Production

For production use, **replace bootstrap samples with actual recorded voice samples** for best quality.

See `voices/README.md` for detailed instructions.

