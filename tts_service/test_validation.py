#!/usr/bin/env python3
"""
TTS Service Test Script

Tests the TTS service and validates output quality.
"""

import asyncio
import base64
import io
import sys
import argparse

try:
    import httpx
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install httpx soundfile numpy")
    sys.exit(1)


class TTSValidator:
    """Test and validate TTS service output"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def test_health(self) -> bool:
        """Test service health"""
        print("1. Checking service health...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            print(f"   ✅ Service healthy")
            print(f"   Model: {health.get('model', 'unknown')}")
            print(f"   Device: {health.get('device', 'unknown')}")
            print(f"   Voices: {', '.join(health.get('voices', []))}")
            return True
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
    
    def validate_audio(self, audio_bytes: bytes) -> dict:
        """Validate audio quality"""
        results = {
            "valid": True,
            "metrics": {},
            "warnings": [],
            "errors": [],
        }
        
        try:
            # Load audio
            buffer = io.BytesIO(audio_bytes)
            audio, sr = sf.read(buffer)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Metrics
            duration = len(audio) / sr
            max_amp = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))
            
            results["metrics"] = {
                "duration_seconds": round(duration, 2),
                "sample_rate": int(sr),
                "max_amplitude": round(float(max_amp), 4),
                "rms_amplitude": round(float(rms), 4),
            }
            
            # Validation checks
            if max_amp < 0.01:
                results["errors"].append("Audio appears silent")
                results["valid"] = False
            elif max_amp < 0.05:
                results["warnings"].append("Audio is very quiet")
            
            if duration < 0.1:
                results["errors"].append("Audio too short")
                results["valid"] = False
            
            # Check for clipping
            clipping = np.sum(np.abs(audio) > 0.95) / len(audio)
            if clipping > 0.01:
                results["warnings"].append(f"Clipping detected: {clipping*100:.1f}%")
            
            # Check variance (constant signal = noise/silence)
            variance = np.var(audio)
            if variance < 1e-6:
                results["errors"].append("Audio is constant (noise/silence)")
                results["valid"] = False
            
        except Exception as e:
            results["errors"].append(f"Audio validation error: {e}")
            results["valid"] = False
        
        return results
    
    async def test_synthesis(
        self,
        text: str,
        language: str = "ar",
        voice_id: str = "reference_voice",
        save_path: str = None,
    ) -> dict:
        """Test speech synthesis"""
        print(f"\n2. Testing synthesis...")
        print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"   Language: {language}")
        print(f"   Voice: {voice_id}")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/synthesize",
                json={
                    "text": text,
                    "language": language,
                    "voice_id": voice_id,
                },
            )
            response.raise_for_status()
            result = response.json()
            
            duration = result.get("duration_seconds", 0)
            print(f"   ✅ Synthesis complete ({duration:.2f}s)")
            
            # Decode and validate audio
            audio_b64 = result.get("audio_base64", "")
            audio_bytes = base64.b64decode(audio_b64)
            
            print("\n3. Validating audio quality...")
            validation = self.validate_audio(audio_bytes)
            
            metrics = validation["metrics"]
            print(f"   Duration: {metrics.get('duration_seconds', 0):.2f}s")
            print(f"   Max Amplitude: {metrics.get('max_amplitude', 0):.4f}")
            print(f"   RMS Amplitude: {metrics.get('rms_amplitude', 0):.4f}")
            
            if validation["warnings"]:
                print(f"\n   ⚠️  Warnings:")
                for w in validation["warnings"]:
                    print(f"      - {w}")
            
            if validation["errors"]:
                print(f"\n   ❌ Errors:")
                for e in validation["errors"]:
                    print(f"      - {e}")
            
            # Save if requested
            if save_path:
                print(f"\n4. Saving audio to {save_path}...")
                with open(save_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"   ✅ Saved ({len(audio_bytes) / 1024:.1f} KB)")
            
            return {
                "success": validation["valid"],
                "duration": duration,
                "validation": validation,
            }
            
        except httpx.HTTPStatusError as e:
            print(f"   ❌ HTTP Error: {e.response.status_code}")
            try:
                detail = e.response.json().get("detail", e.response.text)
            except:
                detail = e.response.text
            print(f"   Detail: {detail}")
            return {"success": False, "error": detail}
        except Exception as e:
            print(f"   ❌ Synthesis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_test_suite(self, save_audio: bool = False):
        """Run comprehensive test suite"""
        print("=" * 60)
        print("TTS Service Test Suite")
        print("=" * 60)
        
        # Health check
        if not await self.test_health():
            print("\n❌ Service not available. Aborting tests.")
            return
        
        # Test cases
        test_cases = [
            {
                "text": "مرحبا، كيف يمكنني مساعدتك؟",
                "language": "ar",
                "voice_id": "reference_voice",
                "save_path": "test_arabic.wav" if save_audio else None,
            },
            {
                "text": "Hello, how can I help you today?",
                "language": "en",
                "voice_id": "reference_voice",
                "save_path": "test_english.wav" if save_audio else None,
            },
        ]
        
        results = []
        for i, tc in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}/{len(test_cases)}")
            print("=" * 60)
            result = await self.test_synthesis(**tc)
            results.append(result)
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        passed = sum(1 for r in results if r.get("success", False))
        print(f"Passed: {passed}/{len(results)}")
        
        if passed == len(results):
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check output above for details.")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(description="Test TTS service")
    parser.add_argument("--url", default="http://localhost:8002", help="Service URL")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--language", default="ar", choices=["ar", "en"])
    parser.add_argument("--voice", default="reference_voice", help="Voice ID")
    parser.add_argument("--save", help="Save audio to file")
    parser.add_argument("--test-suite", action="store_true", help="Run full test suite")
    parser.add_argument("--save-audio", action="store_true", help="Save test audio files")
    
    args = parser.parse_args()
    
    validator = TTSValidator(base_url=args.url)
    
    try:
        if args.test_suite:
            await validator.run_test_suite(save_audio=args.save_audio)
        elif args.text:
            if not await validator.test_health():
                return
            await validator.test_synthesis(
                text=args.text,
                language=args.language,
                voice_id=args.voice,
                save_path=args.save,
            )
        else:
            # Default: quick test
            print("Running quick test (use --test-suite for full tests)\n")
            if not await validator.test_health():
                return
            await validator.test_synthesis(
                text="مرحبا، هذا اختبار.",
                language="ar",
                voice_id="reference_voice",
                save_path=args.save,
            )
    finally:
        await validator.close()


if __name__ == "__main__":
    asyncio.run(main())
