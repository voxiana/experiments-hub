# Quick Start Guide - Voice AI CX Platform

## ✅ System Status

Your gateway and ASR services are now **working correctly**!

## 🚀 Quick Test

### Option 1: Web Interface (Recommended)
1. Open browser: **http://localhost:3001**
2. Click **"Start Call"**
3. Grant microphone permissions
4. Speak into your microphone
5. Watch real-time transcription appear!

### Option 2: API Testing
```bash
# Run the integration test
docker cp test_simple.py voiceai-gateway:/tmp/test_simple.py
docker exec voiceai-gateway python /tmp/test_simple.py
```

## 📊 Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Gateway API | http://localhost:8000 | Main API |
| Gateway Health | http://localhost:8000/health | Health check |
| Gateway Docs | http://localhost:8000/docs | API documentation |
| ASR Service | http://localhost:50051 | Speech-to-text |
| Web Client | http://localhost:3001 | Demo UI |

## 🔧 What Was Fixed

1. **ASR Service URL** - Changed from port 50051 to 8050 (internal)
2. **Environment Variables** - Gateway now reads from docker-compose.yml
3. **Port Configuration** - Fixed ASR server port mismatch

## 📝 API Examples

### 1. Get Authentication Token
```bash
curl -X POST "http://localhost:8000/auth/token?tenant_id=demo&api_key=demo"
```

### 2. Start a Call
```bash
curl -X POST http://localhost:8000/call/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "language": "auto",
    "voice_id": "arabic_gulf_male"
  }'
```

### 3. WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/CALL_ID');

// Send audio
ws.send(JSON.stringify({
  type: 'audio',
  data: base64AudioData
}));

// Receive transcription
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message.type, message.text);
};
```

## 🎯 Next Steps

1. **Test with real speech** - Use the web client (http://localhost:3001)
2. **Check NLU integration** - Verify intent recognition
3. **Test TTS** - Verify voice responses
4. **Monitor logs** - Watch real-time processing

## 📋 Monitoring

### Check Service Status
```bash
# Gateway logs
docker logs voiceai-gateway --tail 50 -f

# ASR logs
docker logs voiceai-asr --tail 50 -f

# All services
docker-compose ps
```

### Health Checks
```bash
# Gateway
curl http://localhost:8000/health

# ASR Service
curl http://localhost:50051/health
```

## 🐛 Troubleshooting

### Gateway not responding
```bash
docker restart voiceai-gateway
docker logs voiceai-gateway
```

### ASR not transcribing
```bash
docker logs voiceai-asr --tail 100
# Check if model is loaded
curl http://localhost:50051/health
```

### WebSocket connection fails
- Make sure call was created first via `/call/start`
- Check browser console for errors
- Verify microphone permissions granted

## 📚 Documentation

- **Full Fix Summary**: See `GATEWAY_FIX_SUMMARY.md`
- **Architecture**: See `docs/architecture.md`
- **Gateway README**: See `gateway/README.md`
- **ASR README**: See `asr_service/README.md`

## ✨ Features Working

- ✅ Call creation and management
- ✅ JWT authentication
- ✅ WebSocket audio streaming
- ✅ ASR transcription (Whisper large-v3)
- ✅ Voice Activity Detection (VAD)
- ✅ Redis pub/sub events
- ✅ Prometheus metrics
- ✅ Health checks

## 🎤 Test Your Voice

The easiest way to test is the web interface:

1. Go to http://localhost:3001
2. Select language (English/Arabic/Auto)
3. Click "Start Call"
4. Speak clearly into your microphone
5. Watch the transcription appear in real-time!

**Note:** The ASR service uses Whisper large-v3 model which provides excellent accuracy for both English and Arabic speech.

---

**System is ready! Start testing with the web client at http://localhost:3001** 🎉

