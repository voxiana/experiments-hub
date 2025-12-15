#!/usr/bin/env python3
"""
Test WebM audio transcription
Creates a test WebM file and sends it to ASR service
"""

import asyncio
import subprocess
import tempfile
import os
import httpx


async def test_webm_transcription():
    """Test ASR service with WebM audio"""
    
    print("="*60)
    print("WebM AUDIO TRANSCRIPTION TEST")
    print("="*60)
    
    # Step 1: Create a test WAV file with ffmpeg
    print("\n1. Creating test audio file...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    
    # Generate 2 seconds of silence (easier to test than sine wave)
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', 'anullsrc=r=16000:cl=mono',
        '-t', '2',
        '-y', wav_path
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ Failed to create test WAV: {result.stderr.decode()}")
        return False
    
    print(f"   ✅ Created test WAV: {wav_path}")
    print(f"   Size: {os.path.getsize(wav_path)} bytes")
    
    # Step 2: Convert WAV to WebM
    print("\n2. Converting WAV to WebM...")
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
        webm_path = tmp_webm.name
    
    cmd = [
        'ffmpeg', '-i', wav_path,
        '-c:a', 'libopus',  # Use Opus codec (common for WebM)
        '-b:a', '64k',
        '-y', webm_path
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ Failed to create WebM: {result.stderr.decode()}")
        os.unlink(wav_path)
        return False
    
    print(f"   ✅ Created WebM file: {webm_path}")
    print(f"   Size: {os.path.getsize(webm_path)} bytes")
    
    # Step 3: Send WebM to ASR service
    print("\n3. Sending WebM to ASR service...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(webm_path, 'rb') as f:
                files = {"file": ("test.webm", f, "audio/webm")}
                response = await client.post(
                    "http://asr-service:8050/transcribe",
                    files=files,
                    params={"language": "auto"}
                )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Transcription successful!")
                print(f"   Text: '{result.get('text', 'N/A')}'")
                print(f"   Language: {result.get('language', 'N/A')}")
                print(f"   Duration: {result.get('duration_seconds', 0):.2f}s")
                print(f"   Inference Time: {result.get('inference_time_seconds', 0):.3f}s")
                print(f"   Load Method: {result.get('load_method', 'N/A')}")
                
                # Clean up
                os.unlink(wav_path)
                os.unlink(webm_path)
                
                print("\n" + "="*60)
                print("✅ WebM TRANSCRIPTION TEST PASSED!")
                print("="*60)
                print("\nThe ASR service can now handle WebM audio from the browser!")
                return True
            else:
                print(f"   ❌ Transcription failed")
                print(f"   Response: {response.text}")
                
                # Clean up
                os.unlink(wav_path)
                os.unlink(webm_path)
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if os.path.exists(webm_path):
            os.unlink(webm_path)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_webm_transcription())
    exit(0 if success else 1)

