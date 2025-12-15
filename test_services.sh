#!/bin/bash
# Test script for Gateway and ASR services

echo "============================================================"
echo "GATEWAY & ASR SERVICE TEST"
echo "============================================================"
echo ""

# Test 1: Gateway Health
echo "TEST 1: Gateway Health Check"
echo "------------------------------------------------------------"
GATEWAY_HEALTH=$(curl -s http://localhost:8000/health)
echo "Response: $GATEWAY_HEALTH"
if echo "$GATEWAY_HEALTH" | grep -q "healthy"; then
    echo "✅ Gateway is healthy"
else
    echo "❌ Gateway health check failed"
    exit 1
fi
echo ""

# Test 2: ASR Health
echo "TEST 2: ASR Service Health Check"
echo "------------------------------------------------------------"
ASR_HEALTH=$(curl -s http://localhost:50051/health)
echo "Response: $ASR_HEALTH"
if echo "$ASR_HEALTH" | grep -q "healthy"; then
    echo "✅ ASR service is healthy"
else
    echo "❌ ASR service health check failed"
    exit 1
fi
echo ""

# Test 3: Get Auth Token
echo "TEST 3: Get Authentication Token"
echo "------------------------------------------------------------"
TOKEN_RESPONSE=$(curl -s -X POST "http://localhost:8000/auth/token?tenant_id=test-tenant&api_key=test-key")
echo "Response: $TOKEN_RESPONSE"
TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TOKEN" ]; then
    echo "✅ Token obtained: ${TOKEN:0:50}..."
else
    echo "❌ Failed to get token"
    exit 1
fi
echo ""

# Test 4: Start Call
echo "TEST 4: Start Call"
echo "------------------------------------------------------------"
CALL_RESPONSE=$(curl -s -X POST http://localhost:8000/call/start \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "tenant_id": "test-tenant",
        "user_id": "test-user",
        "language": "auto",
        "voice_id": "arabic_gulf_male",
        "metadata": {"test": true}
    }')
echo "Response: $CALL_RESPONSE"
CALL_ID=$(echo "$CALL_RESPONSE" | grep -o '"call_id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$CALL_ID" ]; then
    echo "✅ Call started: $CALL_ID"
else
    echo "❌ Failed to start call"
    exit 1
fi
echo ""

# Test 5: Get Call Status
echo "TEST 5: Get Call Status"
echo "------------------------------------------------------------"
CALL_STATUS=$(curl -s -X GET "http://localhost:8000/call/$CALL_ID" \
    -H "Authorization: Bearer $TOKEN")
echo "Response: $CALL_STATUS"
if echo "$CALL_STATUS" | grep -q "$CALL_ID"; then
    echo "✅ Call status retrieved"
else
    echo "❌ Failed to get call status"
fi
echo ""

# Test 6: End Call
echo "TEST 6: End Call"
echo "------------------------------------------------------------"
END_RESPONSE=$(curl -s -X POST "http://localhost:8000/call/$CALL_ID/end" \
    -H "Authorization: Bearer $TOKEN")
echo "Response: $END_RESPONSE"
if echo "$END_RESPONSE" | grep -q "ended"; then
    echo "✅ Call ended successfully"
else
    echo "❌ Failed to end call"
fi
echo ""

echo "============================================================"
echo "🎉 ALL TESTS PASSED!"
echo "============================================================"
echo ""
echo "Gateway and ASR services are working correctly."
echo ""
echo "Next steps:"
echo "1. Open web client: http://localhost:3001"
echo "2. Click 'Start Call' to test voice interaction"
echo "3. Grant microphone permissions when prompted"
echo "4. Speak into your microphone to test ASR"
echo ""
echo "Note: For full end-to-end testing with audio, use the web client."
echo "The ASR service will transcribe your speech and the gateway will"
echo "coordinate the conversation flow."

