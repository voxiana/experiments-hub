# Docker Build Network Error Fix

## Issue
Docker build fails with network errors when trying to download PyTorch:
```
Temporary failure in name resolution
Name or service not known
Failed to establish a new connection
```

## ⚠️ Important Note
**Your ASR service is currently WORKING!** You only need to rebuild if you're making code changes. The running container is fine.

## Quick Check
```bash
# Verify ASR service is running and working
docker ps | grep asr
docker logs voiceai-asr --tail 20
curl http://localhost:50051/health
```

If it's working, **you don't need to rebuild!**

---

## Solutions (Only if you need to rebuild)

### Solution 1: Fix Docker Network/DNS Settings

#### On Windows (Docker Desktop):
1. Open Docker Desktop
2. Go to Settings → Resources → Network
3. Try these DNS servers:
   - Primary: `8.8.8.8` (Google DNS)
   - Secondary: `8.8.4.4`
   - Or: `1.1.1.1` (Cloudflare DNS)
4. Click "Apply & Restart"

#### Alternative: Restart Docker Desktop
Sometimes simply restarting Docker Desktop fixes network issues:
1. Right-click Docker Desktop icon in system tray
2. Select "Quit Docker Desktop"
3. Start Docker Desktop again
4. Wait for it to fully start
5. Try building again

### Solution 2: Use Host Network for Build

Build with host network to bypass Docker's internal network:

```bash
# Stop the current container first (only if rebuilding)
docker stop voiceai-asr

# Build with host network
docker build --network=host -t experiments-hub-asr-service ./asr_service

# Restart the service
docker-compose up -d asr-service
```

### Solution 3: Use Buildkit with Better Caching

Enable Docker Buildkit for better caching and network handling:

```powershell
# Windows PowerShell
$env:DOCKER_BUILDKIT=1
docker-compose build asr-service
```

### Solution 4: Build During Better Network Conditions

Network issues might be temporary (ISP issues, DNS problems, etc.):
- Try building at a different time
- Check if you can access https://pypi.org in your browser
- Check if your internet connection is stable
- Try disconnecting/reconnecting VPN if you use one

### Solution 5: Use Docker Compose Network

Sometimes using docker-compose build works better than direct docker build:

```bash
# This uses docker-compose's network configuration
docker-compose build --no-cache asr-service
```

### Solution 6: Pre-download and Use Local Cache

If network is consistently problematic, you can pre-download packages:

1. Create a local pip cache directory
2. Download packages manually
3. Mount the cache during build

```bash
# Create cache directory
mkdir -p ~/.cache/pip

# Build with cache mounted
docker build --network=host -v ~/.cache/pip:/root/.cache/pip -t experiments-hub-asr-service ./asr_service
```

---

## Current Status Summary

✅ **ASR Service**: Running and working (tested with WebM transcription)  
✅ **Gateway Service**: Running and can connect to ASR  
✅ **WebM Audio**: Successfully processed via ffmpeg conversion  
✅ **All Tests**: Passed

### What's Working:
- Call creation and management
- ASR transcription (including WebM from browser)
- Gateway → ASR connectivity
- Health checks
- Audio format conversion (WAV, WebM, MP3, etc.)

### You Can Now:
1. Use the web client at http://localhost:3001
2. Start calls via API
3. Send audio for transcription
4. Test with real microphone input

---

## When Do You Need to Rebuild?

You only need to rebuild the ASR container if:
- ✅ You modify `asr_service/server.py` or `asr_service/run.py`
- ✅ You change `asr_service/Dockerfile`
- ✅ You update `asr_service/requirements.txt`
- ❌ **NOT** for testing (current container works!)
- ❌ **NOT** for configuration changes (use environment variables)

---

## Troubleshooting Network Issues

### Check Docker Network:
```bash
# Test if Docker can reach the internet
docker run --rm alpine ping -c 4 8.8.8.8
docker run --rm alpine nslookup pypi.org
```

### Check DNS Resolution:
```bash
# Windows
nslookup pypi.org
nslookup files.pythonhosted.org
```

### Check Firewall/Antivirus:
- Temporarily disable antivirus/firewall
- Check if Docker is allowed through firewall
- Check Windows Defender settings

### Check Proxy Settings:
If you're behind a corporate proxy:
1. Docker Desktop → Settings → Resources → Proxies
2. Configure HTTP/HTTPS proxy settings
3. Add proxy to pip config in Dockerfile

---

## Summary

**Your system is working!** The build error is a network issue, not a code issue. 

The ASR service is running successfully and can:
- ✅ Process WebM audio from browsers
- ✅ Transcribe speech using Whisper large-v3
- ✅ Handle multiple audio formats
- ✅ Connect with the gateway

**Only rebuild if you're making code changes**, and only then do you need to address the network issue.

For testing and using the system, **you're all set!** 🎉

