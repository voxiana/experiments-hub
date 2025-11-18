# Web Client

## Overview

The Web Client provides a demo web interface for the Voice AI CX Platform. It's a lightweight, browser-based client that demonstrates real-time voice conversations with the AI system using WebRTC audio capture and WebSocket streaming.

## Features

- **Voice Input**: WebRTC-based audio capture from microphone
- **Real-time Streaming**: WebSocket connection for bi-directional audio
- **Live Transcription**: Display of interim and final transcripts
- **Audio Visualization**: Waveform visualization during recording
- **Language Selection**: Choose between Arabic, English, or Auto-detect
- **Barge-in Support**: Interrupt bot responses
- **Simple UI**: Clean, minimal interface

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Web Client                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │  WebRTC    │  │  WebSocket  │  │     UI       │      │
│  │   Audio    │  │   Client    │  │  Components  │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │   Gateway   │                        │
│                    │   Service   │                        │
│                    └─────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Technology Stack

- **HTML5** - Structure
- **CSS3** - Styling
- **Vanilla JavaScript** - No frameworks
- **WebRTC** - Audio capture (getUserMedia)
- **WebSocket** - Real-time communication
- **Base64** - Audio encoding

## Files

```
web/
├── index.html      # Main web client (all-in-one)
├── Dockerfile      # Nginx container
└── README.md       # This file
```

## Usage

### Local Development

Simply open `index.html` in a modern browser:

```bash
# Open directly
open index.html

# Or serve via Python
python3 -m http.server 3001
# Access at http://localhost:3001
```

### Docker Deployment

```bash
docker build -t web-client:latest .
docker run -p 3001:80 web-client:latest
```

### Docker Compose

From repository root:

```bash
docker-compose up web
# Access at http://localhost:3001
```

## Configuration

### Gateway URL

Update the gateway URL in `index.html`:

```javascript
const GATEWAY_URL = 'http://localhost:8000';  // Update for production
```

### Language Options

Available languages:
- **Arabic** (ar) - Arabic voice and ASR
- **English** (en) - English voice and ASR
- **Auto** (auto) - Automatic language detection

## Features Walkthrough

### 1. Start Call

Click "Start Call" to:
1. Request microphone permission
2. Connect to gateway via REST API (`POST /call/start`)
3. Establish WebSocket connection
4. Begin audio streaming

### 2. Voice Input

- **Push to Talk**: Hold the "Speak" button while talking
- **Audio Capture**: 16kHz mono PCM audio
- **Chunk Size**: 250ms chunks sent to gateway
- **Visualization**: Real-time waveform display

### 3. Transcription Display

- **Interim Transcripts**: Gray text (partial results)
- **Final Transcripts**: Bold black text (confirmed)
- **Bot Responses**: Green text with audio playback

### 4. Barge-in

- Release "Speak" button to interrupt bot
- Sends `barge_in` event to stop TTS playback

### 5. End Call

Click "End Call" to:
1. Close WebSocket connection
2. Stop audio capture
3. Release microphone

## WebSocket Protocol

### Client → Server

**Audio Chunk**:
```json
{
  "type": "audio",
  "data": "<base64 PCM audio>"
}
```

**Barge-in**:
```json
{
  "type": "barge_in"
}
```

**Ping** (keep-alive):
```json
{
  "type": "ping"
}
```

### Server → Client

**Interim Transcript**:
```json
{
  "type": "transcript_interim",
  "text": "Hello, how...",
  "timestamp": 1699999999.123
}
```

**Final Transcript**:
```json
{
  "type": "transcript_final",
  "text": "Hello, how can I help you?",
  "timestamp": 1699999999.456
}
```

**Bot Response**:
```json
{
  "type": "response",
  "text": "I'd be happy to help you today.",
  "audio": "<base64 WAV audio>",
  "timestamp": 1700000000.789
}
```

## Browser Compatibility

### Supported Browsers

- ✅ Chrome 80+ (recommended)
- ✅ Firefox 75+
- ✅ Safari 14+
- ✅ Edge 80+

### Required Features

- WebRTC (getUserMedia)
- WebSocket
- Audio API
- ES6+ JavaScript

## Customization

### Styling

Modify CSS in `<style>` section:

```css
.btn-primary {
  background-color: #007bff;  /* Change button color */
}

.transcript-bot {
  color: #28a745;  /* Change bot text color */
}
```

### Audio Settings

Adjust audio capture settings:

```javascript
const constraints = {
  audio: {
    sampleRate: 16000,
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
};
```

## Troubleshooting

### Microphone Not Working

**Problem**: "Permission denied" or no audio capture

**Solutions**:
- Grant microphone permission in browser
- Use HTTPS (required for mic access except localhost)
- Check browser privacy settings
- Verify microphone is not used by another app

---

### WebSocket Connection Failed

**Problem**: Cannot connect to gateway

**Solutions**:
- Verify gateway is running (`curl http://localhost:8000/health`)
- Check CORS settings in gateway
- Update `GATEWAY_URL` in code
- Check browser console for errors

---

### No Audio Playback

**Problem**: Bot responses don't play audio

**Solutions**:
- Check browser autoplay policy (user interaction required)
- Verify audio element is not muted
- Check base64 audio decoding
- Review browser console for errors

---

### Poor Audio Quality

**Problem**: Distorted or choppy audio

**Solutions**:
- Check internet connection
- Reduce audio chunk size
- Verify sample rate matches (16kHz)
- Check CPU usage (close other apps)

## Development

### Adding Features

**Call Recording**:
```javascript
const chunks = [];
mediaRecorder.ondataavailable = e => chunks.push(e.data);
mediaRecorder.onstop = () => {
  const blob = new Blob(chunks, {type: 'audio/wav'});
  // Save or upload blob
};
```

**Sentiment Display**:
```javascript
// Add to UI when receiving bot response
if (data.sentiment) {
  showSentiment(data.sentiment);  // Display emoji or indicator
}
```

## Production Considerations

### Security

- **HTTPS**: Required for microphone access
- **Authentication**: Add JWT token to WebSocket
- **CORS**: Configure allowed origins in gateway
- **CSP**: Set Content-Security-Policy headers

### Performance

- **Audio Compression**: Consider using Opus codec
- **Chunk Buffering**: Buffer audio chunks to reduce latency
- **Connection Pooling**: Reuse WebSocket connections
- **Error Handling**: Gracefully handle disconnections

### Deployment

**Nginx Configuration**:
```nginx
server {
  listen 80;
  server_name example.com;

  location / {
    root /usr/share/nginx/html;
    index index.html;
  }

  # WebSocket proxy (if needed)
  location /ws/ {
    proxy_pass http://gateway:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
}
```

## Mobile Support

### Responsive Design

The UI is mobile-friendly with responsive breakpoints:

```css
@media (max-width: 768px) {
  .container {
    max-width: 100%;
    padding: 1rem;
  }
}
```

### Mobile Browsers

- iOS Safari: Requires user interaction before mic access
- Android Chrome: Full WebRTC support
- Mobile Firefox: WebRTC supported

### Touch Gestures

Consider adding:
- Touch and hold for push-to-talk
- Swipe gestures for navigation
- Haptic feedback on actions

## Best Practices

1. **Error Handling**: Always handle microphone and WebSocket errors
2. **User Feedback**: Show loading states and connection status
3. **Accessibility**: Add ARIA labels for screen readers
4. **Performance**: Debounce rapid button clicks
5. **Privacy**: Show mic indicator when recording
6. **Testing**: Test across multiple browsers and devices

## Alternatives

### Framework-based Clients

For production applications, consider:
- **React** - Component-based UI
- **Vue.js** - Reactive framework
- **Angular** - Full-featured framework

### Native Mobile Apps

For better mobile experience:
- **React Native** - Cross-platform mobile
- **Flutter** - Native performance
- **Swift/Kotlin** - Native iOS/Android

## Roadmap

- [ ] Video support (camera input)
- [ ] Screen sharing
- [ ] Multi-language UI
- [ ] Dark mode
- [ ] Call history
- [ ] Settings panel (voice selection, volume)
- [ ] Offline support (PWA)
- [ ] Push notifications

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- WebRTC: [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
