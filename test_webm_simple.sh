#!/bin/bash
# Simple WebM transcription test

echo "============================================================"
echo "WebM AUDIO TRANSCRIPTION TEST"
echo "============================================================"

# Create test WAV file (2 seconds of silence)
echo ""
echo "1. Creating test WAV file..."
ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 2 -y /tmp/test.wav 2>&1 | grep -v "^frame="
echo "   ✅ Created /tmp/test.wav"

# Convert to WebM
echo ""
echo "2. Converting to WebM..."
ffmpeg -i /tmp/test.wav -c:a libopus -b:a 64k -y /tmp/test.webm 2>&1 | grep -v "^frame="
echo "   ✅ Created /tmp/test.webm"
ls -lh /tmp/test.webm

# Send to ASR service
echo ""
echo "3. Sending WebM to ASR service..."
RESPONSE=$(curl -s -X POST http://localhost:8050/transcribe \
    -F "file=@/tmp/test.webm" \
    -F "language=auto")

echo "   Response: $RESPONSE"

# Check if successful
if echo "$RESPONSE" | grep -q '"text"'; then
    echo ""
    echo "============================================================"
    echo "✅ WebM TRANSCRIPTION TEST PASSED!"
    echo "============================================================"
    echo ""
    echo "The ASR service successfully processed WebM audio!"
    echo "It can now handle audio from the web browser."
    exit 0
else
    echo ""
    echo "============================================================"
    echo "❌ WebM TRANSCRIPTION TEST FAILED"
    echo "============================================================"
    exit 1
fi

