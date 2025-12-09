#!/bin/bash
set -e

echo "=========================================="
echo "TTS Service Entrypoint"
echo "=========================================="

# Ensure voices directory exists
VOICES_DIR="/app/voices"
mkdir -p "$VOICES_DIR"

# Check for reference_voice.wav (minimum requirement)
if [ ! -f "$VOICES_DIR/reference_voice.wav" ]; then
    echo ""
    echo "⚠️  WARNING: reference_voice.wav not found!"
    echo ""
    echo "The TTS service requires at least one valid voice sample."
    echo "Please add a WAV file (24kHz, mono, 6-12s of speech) to:"
    echo "  $VOICES_DIR/reference_voice.wav"
    echo ""
    echo "Running bootstrap to create placeholders..."
    python3 /app/bootstrap_voices.py || true
    echo ""
fi

# List available voice files
echo "Voice files in $VOICES_DIR:"
for f in "$VOICES_DIR"/*.wav; do
    if [ -f "$f" ]; then
        basename "$f"
    fi
done
echo ""

echo "Starting TTS service..."
exec "$@"
