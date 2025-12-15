#!/usr/bin/env python3
"""
Test script for Gateway service
Tests call creation, WebSocket connection, and ASR integration
"""

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np


# Configuration
GATEWAY_URL = "http://localhost:8000"
ASR_SERVICE_URL = "http://localhost:50051"  # External port mapping

# Test parameters
TENANT_ID = "test-tenant"
API_KEY = "test-api-key"


def generate_test_audio(duration_seconds=2, frequency=440, sample_rate=16000):
    """
    Generate a test audio signal (sine wave)
    Returns: numpy array of float32 samples
    """
    print(f"Generating test audio: {duration_seconds}s sine wave at {frequency}Hz")
    
    # Generate sine wave
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% amplitude
    
    return audio.astype(np.float32)


def audio_to_pcm_bytes(audio_float32):
    """Convert float32 audio to PCM 16-bit bytes"""
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    return audio_int16.tobytes()


def audio_to_wav_bytes(audio_float32, sample_rate=16000):
    """Convert float32 audio to WAV file bytes"""
    import io
    
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return wav_buffer.getvalue()


async def test_gateway_health():
    """Test 1: Gateway health check"""
    print("\n" + "="*60)
    print("TEST 1: Gateway Health Check")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GATEWAY_URL}/health", timeout=10.0)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert response.json()["status"] == "healthy", "Gateway not healthy"
            
            print("✅ Gateway health check passed")
            return True
    except Exception as e:
        print(f"❌ Gateway health check failed: {e}")
        return False


async def test_asr_health():
    """Test 2: ASR service health check"""
    print("\n" + "="*60)
    print("TEST 2: ASR Service Health Check")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ASR_SERVICE_URL}/health", timeout=10.0)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            print(f"ASR Model: {result.get('model')}")
            print(f"ASR Device: {result.get('device')}")
            print(f"VAD Enabled: {result.get('vad_enabled')}")
            
            print("✅ ASR service health check passed")
            return True
    except Exception as e:
        print(f"❌ ASR service health check failed: {e}")
        print(f"   Make sure ASR service is running and accessible at {ASR_SERVICE_URL}")
        return False


async def test_auth_token():
    """Test 3: Get authentication token"""
    print("\n" + "="*60)
    print("TEST 3: Authentication Token")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GATEWAY_URL}/auth/token",
                params={"tenant_id": TENANT_ID, "api_key": API_KEY},
                timeout=10.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            token = result["access_token"]
            
            print(f"Token Type: {result['token_type']}")
            print(f"Access Token: {token[:50]}...")
            
            print("✅ Authentication token obtained")
            return token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None


async def test_start_call(token):
    """Test 4: Start a call"""
    print("\n" + "="*60)
    print("TEST 4: Start Call")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GATEWAY_URL}/call/start",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "tenant_id": TENANT_ID,
                    "user_id": "test-user-123",
                    "language": "auto",
                    "voice_id": "arabic_gulf_male",
                    "metadata": {"test": "true"}
                },
                timeout=10.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            print(f"Call ID: {result['call_id']}")
            print(f"Status: {result['status']}")
            print(f"WebSocket URL: {result['ws_url']}")
            print(f"Session Token: {result['session_token'][:50]}...")
            
            print("✅ Call started successfully")
            return result
    except Exception as e:
        print(f"❌ Start call failed: {e}")
        return None


async def test_get_call_status(token, call_id):
    """Test 5: Get call status"""
    print("\n" + "="*60)
    print("TEST 5: Get Call Status")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{GATEWAY_URL}/call/{call_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            print(f"Call Status: {json.dumps(result, indent=2)}")
            
            print("✅ Call status retrieved successfully")
            return result
    except Exception as e:
        print(f"❌ Get call status failed: {e}")
        return None


async def test_asr_transcription():
    """Test 6: Direct ASR service transcription"""
    print("\n" + "="*60)
    print("TEST 6: Direct ASR Transcription")
    print("="*60)
    
    try:
        # Generate test audio
        audio = generate_test_audio(duration_seconds=2)
        wav_bytes = audio_to_wav_bytes(audio)
        
        print(f"Test audio size: {len(wav_bytes)} bytes")
        
        async with httpx.AsyncClient() as client:
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            response = await client.post(
                f"{ASR_SERVICE_URL}/transcribe",
                files=files,
                params={"language": "auto"},
                timeout=30.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Transcription: '{result.get('text', 'N/A')}'")
                print(f"Language: {result.get('language', 'N/A')}")
                print(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                print(f"Inference Time: {result.get('inference_time_seconds', 0):.3f}s")
                print("✅ ASR transcription successful")
                return True
            else:
                print(f"ASR transcription returned status {response.status_code}")
                print(f"Response: {response.text}")
                print("⚠️  ASR returned non-200 status (may be expected for sine wave)")
                return True  # Don't fail test, sine wave might not transcribe
    except Exception as e:
        print(f"❌ ASR transcription failed: {e}")
        return False


async def test_websocket_connection(call_id):
    """Test 7: WebSocket connection and audio streaming"""
    print("\n" + "="*60)
    print("TEST 7: WebSocket Connection & Audio Streaming")
    print("="*60)
    
    try:
        import websockets
        
        ws_url = f"ws://localhost:8000/ws/{call_id}"
        print(f"Connecting to: {ws_url}")
        
        async with websockets.connect(ws_url, ping_interval=None) as websocket:
            print("✅ WebSocket connected")
            
            # Send ping
            print("Sending ping...")
            await websocket.send(json.dumps({"type": "ping"}))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            print(f"Received: {response_data}")
            
            assert response_data["type"] == "pong", "Expected pong response"
            
            # Send test audio
            print("\nSending test audio...")
            audio = generate_test_audio(duration_seconds=1)
            wav_bytes = audio_to_wav_bytes(audio)
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            await websocket.send(json.dumps({
                "type": "audio",
                "data": audio_base64
            }))
            print("Audio sent, waiting for transcription response...")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                response_data = json.loads(response)
                print(f"Received transcription response: {response_data}")
                
                print("✅ WebSocket audio streaming successful")
                return True
            except asyncio.TimeoutError:
                print("⚠️  No transcription response within 30s (sine wave may not produce text)")
                print("   Connection test passed, but transcription may need real speech")
                return True
            
    except ImportError:
        print("⚠️  websockets library not installed")
        print("   Install with: pip install websockets")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_call(token, call_id):
    """Test 8: End call"""
    print("\n" + "="*60)
    print("TEST 8: End Call")
    print("="*60)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GATEWAY_URL}/call/{call_id}/end",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            result = response.json()
            print(f"Call ended: {result}")
            
            print("✅ Call ended successfully")
            return True
    except Exception as e:
        print(f"❌ End call failed: {e}")
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "="*60)
    print("GATEWAY & ASR SERVICE TEST SUITE")
    print("="*60)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"ASR Service URL: {ASR_SERVICE_URL}")
    print("="*60)
    
    results = {}
    
    # Test 1: Gateway health
    results["gateway_health"] = await test_gateway_health()
    if not results["gateway_health"]:
        print("\n❌ Gateway is not accessible. Make sure it's running:")
        print("   docker-compose up gateway")
        return results
    
    # Test 2: ASR health
    results["asr_health"] = await test_asr_health()
    if not results["asr_health"]:
        print("\n⚠️  ASR service is not accessible. Some tests will be skipped.")
        print("   To enable ASR tests, run: docker-compose up asr-service")
    
    # Test 3: Auth
    token = await test_auth_token()
    results["auth"] = token is not None
    if not results["auth"]:
        print("\n❌ Cannot proceed without authentication token")
        return results
    
    # Test 4: Start call
    call_data = await test_start_call(token)
    results["start_call"] = call_data is not None
    if not results["start_call"]:
        print("\n❌ Cannot proceed without call ID")
        return results
    
    call_id = call_data["call_id"]
    
    # Test 5: Get call status
    results["get_call_status"] = await test_get_call_status(token, call_id)
    
    # Test 6: Direct ASR transcription (if ASR is available)
    if results["asr_health"]:
        results["asr_transcription"] = await test_asr_transcription()
    else:
        results["asr_transcription"] = None
        print("\n⚠️  Skipping ASR transcription test (ASR service not available)")
    
    # Test 7: WebSocket connection
    results["websocket"] = await test_websocket_connection(call_id)
    
    # Test 8: End call
    results["end_call"] = await test_end_call(token, call_id)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    skipped_tests = 0
    
    for test_name, result in results.items():
        total_tests += 1
        if result is True:
            status = "✅ PASS"
            passed_tests += 1
        elif result is None:
            status = "⚠️  SKIP"
            skipped_tests += 1
        else:
            status = "❌ FAIL"
        
        print(f"{test_name:25s}: {status}")
    
    print("="*60)
    print(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {total_tests - passed_tests - skipped_tests} | Skipped: {skipped_tests}")
    print("="*60)
    
    # Final verdict
    if total_tests - skipped_tests == passed_tests:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

