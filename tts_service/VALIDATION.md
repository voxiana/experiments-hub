# TTS Output Validation Guide

This guide explains how to validate that the TTS service is generating correct output.

## Quick Validation

### 1. Using the Validation Endpoint

The service provides a `/validate` endpoint that synthesizes audio and checks quality:

```bash
curl -X POST http://localhost:8002/validate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا، كيف يمكنني مساعدتك؟",
    "language": "ar",
    "voice_id": "arabic_gulf_male"
  }'
```

Response includes:
- `valid`: Boolean indicating if output passes validation
- `metrics`: Audio quality metrics (duration, SNR, clipping, etc.)
- `warnings`: Non-critical issues
- `errors`: Critical problems
- `audio_base64`: The generated audio (for manual verification)

### 2. Using the Test Script

Run the validation test script:

```bash
# Basic test
python test_validation.py --text "Hello, this is a test"

# Test with Arabic
python test_validation.py --text "مرحبا" --language ar

# Save audio output
python test_validation.py --text "Test" --save output.wav

# Run comprehensive test suite
python test_validation.py --test-suite

# Test against different service URL
python test_validation.py --url http://localhost:8002 --text "Test"
```

### 3. Manual Validation

#### Step 1: Synthesize Audio

```bash
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your test text here",
    "language": "ar",
    "voice_id": "arabic_gulf_male"
  }' | jq -r '.audio_base64' | base64 -d > output.wav
```

#### Step 2: Play and Listen

```bash
# Play the audio (Linux/Mac)
aplay output.wav  # or
afplay output.wav  # macOS

# Or use any audio player
```

#### Step 3: Verify Content

Listen to the audio and verify:
- ✅ Words match the input text
- ✅ Language is correct
- ✅ Voice sounds natural (not robotic)
- ✅ No noise or distortion
- ✅ Appropriate speed and pacing

## Validation Metrics

The validation endpoint checks:

### Audio Quality Metrics

1. **Duration**: Matches expected duration (within tolerance)
2. **Sample Rate**: Should be 24000 Hz
3. **Amplitude**: 
   - Max amplitude should be > 0.01 (not silent)
   - Max amplitude should be < 1.0 (not clipped)
4. **Signal-to-Noise Ratio (SNR)**: Should be > 10 dB for good quality
5. **Clipping Ratio**: Should be < 1% (no distortion)
6. **Signal Variance**: Should be > 1e-6 (not constant/noise)

### Common Issues and Solutions

#### Issue: Audio is silent or very quiet
- **Check**: `max_amplitude < 0.01`
- **Solution**: Verify reference audio file exists and is valid

#### Issue: Audio contains noise instead of speech
- **Check**: `snr_db < 10` or `signal_variance` is very low
- **Solution**: Ensure reference audio is actual speech, not a tone/noise

#### Issue: Audio is clipped/distorted
- **Check**: `clipping_ratio > 0.01`
- **Solution**: Check audio normalization in the synthesis pipeline

#### Issue: Duration mismatch
- **Check**: Actual duration differs significantly from expected
- **Solution**: Verify text preprocessing and sentence splitting

## Automated Validation

### Using ASR (Speech-to-Text) for Content Verification

For advanced validation, you can use ASR to verify the generated audio contains the correct words:

```python
import httpx
import base64

# 1. Generate TTS audio
tts_response = httpx.post(
    "http://localhost:8002/synthesize",
    json={"text": "Hello world", "language": "en"}
)
audio_b64 = tts_response.json()["audio_base64"]

# 2. Send to ASR service for transcription
asr_response = httpx.post(
    "http://localhost:8001/transcribe",  # ASR service
    files={"audio": base64.b64decode(audio_b64)}
)
transcribed_text = asr_response.json()["text"]

# 3. Compare
expected = "Hello world"
actual = transcribed_text
similarity = calculate_similarity(expected, actual)
print(f"Text match: {similarity:.2%}")
```

## Continuous Validation

### Integration with CI/CD

Add validation to your test suite:

```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_tts_validation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8002/validate",
            json={
                "text": "Test text",
                "language": "en",
                "voice_id": "english_uae_male"
            }
        )
        result = response.json()
        assert result["valid"], f"Validation failed: {result['errors']}"
        assert result["metrics"]["validation_score"] > 0.8
```

## Best Practices

1. **Always validate after deployment** - Run validation tests after service updates
2. **Test with diverse inputs** - Test different languages, text lengths, and voices
3. **Monitor metrics** - Track validation scores over time
4. **Manual spot checks** - Periodically listen to generated audio
5. **Compare with baseline** - Keep reference audio samples for comparison

## Troubleshooting

### Validation Endpoint Returns Errors

1. Check service health: `curl http://localhost:8002/health`
2. Check logs for synthesis errors
3. Verify reference audio files exist in `voices/` directory
4. Check GPU/CPU availability if using GPU acceleration

### Audio Quality Issues

1. **Robotic voice**: Reference audio may be poor quality or too short
2. **Wrong language**: Verify language parameter matches text language
3. **Missing words**: Check text preprocessing and sentence splitting
4. **Speed issues**: Adjust `speed` parameter (0.7-1.5 range)

## Example Validation Workflow

```bash
# 1. Health check
curl http://localhost:8002/health

# 2. Validate Arabic text
python test_validation.py \
  --text "مرحبا، كيف يمكنني مساعدتك؟" \
  --language ar \
  --voice arabic_gulf_male \
  --save test_arabic.wav

# 3. Validate English text
python test_validation.py \
  --text "Hello, how can I help you?" \
  --language en \
  --voice english_uae_male \
  --save test_english.wav

# 4. Run full test suite
python test_validation.py --test-suite

# 5. Listen to generated files
aplay test_arabic.wav
aplay test_english.wav
```

## Validation Checklist

- [ ] Service health check passes
- [ ] Validation endpoint returns `valid: true`
- [ ] No errors in validation response
- [ ] Audio duration matches expected
- [ ] SNR > 10 dB
- [ ] Clipping ratio < 1%
- [ ] Manual listening confirms correct words
- [ ] Voice sounds natural (not robotic)
- [ ] Language is correct
- [ ] Speed/pacing is appropriate

