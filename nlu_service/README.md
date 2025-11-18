# NLU Service

## Overview

The NLU (Natural Language Understanding) Service is the conversational AI engine for the Voice AI CX Platform. It uses large language models (vLLM with Qwen2.5-72B or Llama-3.1-70B) to understand user intent, generate contextually appropriate responses, execute tool calls, and manage multi-turn conversations. The service supports tool calling, RAG integration, and automatic escalation to human agents.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    NLU Service                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │   Intent   │  │   Dialog    │  │   Tool       │      │
│  │Classifier  │  │  Manager    │  │  Executor    │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│         ┌─────────────────┴──────────────────┐            │
│         │                                    │             │
│    ┌────▼────┐  ┌──────────┐  ┌──────────┐ │            │
│    │  vLLM   │  │   RAG    │  │   CRM    │ │            │
│    │ Server  │  │ Service  │  │Connector │ │            │
│    └─────────┘  └──────────┘  └──────────┘ │            │
│         │                                    │             │
│    ┌────▼────┐                               │            │
│    │Langfuse │  (LLM Observability)         │            │
│    └─────────┘                               │            │
└──────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities

- **Intent Classification**: Identifies user intent (greeting, question, complaint, request, escalation, etc.)
- **Multi-turn Dialog Management**: Maintains conversation context across turns
- **Tool Calling**: Executes actions via function calling (search KB, get order status, create ticket, etc.)
- **RAG Integration**: Retrieves relevant information from knowledge base
- **CRM Integration**: Accesses customer data and creates tickets
- **Multilingual Support**: Arabic (MSA & Gulf) and English with code-switching
- **Automatic Escalation**: Policy-based escalation to human agents
- **LLM Observability**: Integration with Langfuse for monitoring
- **Emotion-aware**: Considers detected emotion in responses and escalation logic

### Supported Intents

- `GREETING` - Initial greetings and pleasantries
- `QUESTION` - Information requests
- `COMPLAINT` - Issues and complaints
- `REQUEST` - Action requests (check order, update info, etc.)
- `ESCALATION` - Explicit request for human agent
- `FEEDBACK` - Customer feedback
- `GOODBYE` - Conversation closing
- `UNKNOWN` - Unclear intent

### Available Tools (Function Calling)

1. **search_kb** - Search knowledge base for information
2. **get_order_status** - Retrieve order status
3. **get_customer_info** - Fetch customer information from CRM
4. **create_ticket** - Create support ticket
5. **escalate_to_human** - Escalate to human agent
6. **update_ticket** - Update existing ticket
7. **send_email** - Send email notification
8. **send_sms** - Send SMS notification

## Technology Stack

### Core Framework

- **Python 3.10+** - Runtime environment
- **FastAPI 0.104.1** - API framework
- **Pydantic 2.5.0** - Data validation and models
- **aiohttp 3.9.0** - Async HTTP client

### AI/ML

- **vLLM Server** - High-performance LLM inference
  - Models: Qwen2.5-72B-Instruct or Llama-3.1-70B-Instruct
  - OpenAI-compatible API
- **Transformers 4.36.0** - Model utilities
- **LangChain 0.1.0** - Conversational AI framework

### Observability

- **Langfuse 2.7.0** - LLM observability and tracing
- **OpenTelemetry** - Distributed tracing
- **Prometheus Client** - Metrics collection

### Integration

- **Redis** - Conversation state (via gateway)
- **RAG Service** - Knowledge base queries
- **CRM Connectors** - Customer data and ticketing

## Configuration

### Environment Variables

```bash
# vLLM Server
VLLM_URL="http://vllm:8000/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  # or "meta-llama/Llama-3.1-70B-Instruct"

# RAG Service
RAG_URL="http://rag-service:8080"

# CRM Connectors
CRM_URL="http://connectors:8090"

# Langfuse (LLM Observability)
LANGFUSE_PUBLIC_KEY="pk-..."
LANGFUSE_SECRET_KEY="sk-..."
LANGFUSE_HOST="https://cloud.langfuse.com"

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
OTEL_SERVICE_NAME="nlu"

# Server
HOST="0.0.0.0"
PORT=8001

# Model Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=200
CONVERSATION_HISTORY_LIMIT=20  # Keep last 20 messages
```

## Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export VLLM_URL="http://localhost:8000/v1"
export RAG_URL="http://localhost:8080"
export CRM_URL="http://localhost:8090"
```

3. **Run the service**:
```bash
python policy.py
# Service will start on http://0.0.0.0:8001
```

### Docker Deployment

```bash
docker build -t nlu-service:latest .
docker run -p 8001:8001 \
  -e VLLM_URL="http://vllm:8000/v1" \
  -e RAG_URL="http://rag-service:8080" \
  nlu-service:latest
```

### Docker Compose

From repository root:

```bash
docker-compose up nlu-service
```

## API Reference

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-72B-Instruct"
}
```

---

### Process Conversation Turn

Process a single conversation turn and generate response.

**Endpoint**: `POST /process`

**Request Body**:
```json
{
  "call_id": "call_a1b2c3d4",
  "text": "I need to check my order status",
  "tenant_id": "tenant_123",
  "language": "en",
  "context": {
    "user_id": "user_456",
    "customer_id": "C12345",
    "phone": "+971-50-123-4567"
  }
}
```

**Parameters**:
- `call_id` (required): Unique call/session identifier
- `text` (required): User's message text
- `tenant_id` (required): Tenant identifier
- `language` (default: "en"): Language code (`en`, `ar`, `auto`)
- `context` (optional): Additional customer context

**Response**:
```json
{
  "intent": "request",
  "confidence": 0.9,
  "text_response": "I'd be happy to help you check your order status. Could you please provide your order number?",
  "tool_calls": [],
  "emotion": null,
  "should_escalate": false,
  "metadata": {
    "inference_time_ms": 342.5,
    "tool_results": {},
    "model": "Qwen/Qwen2.5-72B-Instruct"
  }
}
```

**Response with Tool Call**:
```json
{
  "intent": "request",
  "confidence": 0.95,
  "text_response": "Your order #ORD12345 has been shipped via DHL (tracking: DHL123456) and should arrive by November 8th.",
  "tool_calls": [
    {
      "tool": "get_order_status",
      "parameters": {
        "order_id": "ORD12345"
      }
    }
  ],
  "should_escalate": false,
  "metadata": {
    "inference_time_ms": 456.7,
    "tool_results": {
      "get_order_status": {
        "order_id": "ORD12345",
        "status": "shipped",
        "tracking": "DHL123456",
        "eta": "2025-11-08"
      }
    },
    "model": "Qwen/Qwen2.5-72B-Instruct"
  }
}
```

---

### Clear Conversation History

Clear conversation history for a specific call.

**Endpoint**: `POST /clear/{call_id}`

**Response**:
```json
{
  "status": "cleared"
}
```

## Data Models

### NLUResponse

```python
class NLUResponse(BaseModel):
    intent: IntentType              # Classified intent
    confidence: float               # Confidence score (0-1)
    text_response: str              # Generated response text
    tool_calls: List[ToolCall]      # Executed tool calls
    emotion: Optional[str]          # Detected emotion (if available)
    should_escalate: bool           # Escalation flag
    metadata: Dict                  # Additional metadata
```

### ToolCall

```python
class ToolCall(BaseModel):
    tool: ActionType                # Tool/action name
    parameters: Dict[str, Any]      # Tool parameters
```

### Intent Types

```python
class IntentType(str, Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMPLAINT = "complaint"
    REQUEST = "request"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"
```

## Tool Definitions

### search_kb

Search the knowledge base for information.

**Parameters**:
```json
{
  "query": "What are your return policies?",
  "category": "policy"  // optional: product, policy, technical, billing
}
```

**Returns**:
```json
{
  "results": [
    {
      "text": "Items can be returned within 30 days...",
      "source": "Return Policy Documentation",
      "score": 0.92
    }
  ],
  "sources": ["Return Policy Documentation", "FAQ"]
}
```

---

### get_order_status

Get the status of a customer order.

**Parameters**:
```json
{
  "order_id": "ORD12345"
}
```

**Returns**:
```json
{
  "order_id": "ORD12345",
  "status": "shipped",
  "tracking": "DHL123456",
  "eta": "2025-11-08"
}
```

---

### get_customer_info

Retrieve customer information from CRM.

**Parameters**:
```json
{
  "customer_id": "C12345"  // or phone number
}
```

**Returns**:
```json
{
  "customer_id": "C12345",
  "name": "Ahmed Al-Mansoori",
  "tier": "gold",
  "phone": "+971-50-123-4567",
  "email": "ahmed@example.ae"
}
```

---

### create_ticket

Create a support ticket in the CRM system.

**Parameters**:
```json
{
  "subject": "Damaged product received",
  "description": "Customer received damaged item in order #ORD12345",
  "priority": "high"  // low, medium, high, urgent
}
```

**Returns**:
```json
{
  "ticket_id": "TICKET-1699564800",
  "status": "created",
  "subject": "Damaged product received",
  "priority": "high"
}
```

---

### escalate_to_human

Escalate conversation to human agent.

**Parameters**:
```json
{
  "reason": "customer_request",  // or: technical_issue, complaint, etc.
  "urgency": "medium"            // low, medium, high
}
```

**Returns**:
```json
{
  "status": "escalated",
  "reason": "customer_request"
}
```

## Conversation Flow

### Example: Simple Question

**User**: "What are your support hours?"

**Process**:
1. Classify intent → `QUESTION`
2. Tool call → `search_kb(query="support hours")`
3. Execute tool → RAG returns relevant info
4. Generate response with context

**Response**: "Our support team is available Sunday to Thursday, from 9 AM to 6 PM UAE time."

---

### Example: Order Status

**User**: "Can you check my order #ORD12345?"

**Process**:
1. Classify intent → `REQUEST`
2. Tool call → `get_order_status(order_id="ORD12345")`
3. Execute tool → Fetch from CRM
4. Generate response with order details

**Response**: "Your order #ORD12345 has been shipped via DHL (tracking: DHL123456) and should arrive by November 8th."

---

### Example: Escalation

**User**: "I want to speak to a manager!"

**Process**:
1. Classify intent → `ESCALATION`
2. Check escalation keywords
3. Tool call → `escalate_to_human(reason="customer_request", urgency="medium")`
4. Set `should_escalate=true`

**Response**: "I understand. Let me connect you with one of our team members who can better assist you."

---

### Example: Multi-turn Conversation

**Turn 1**:
- User: "Hello"
- Bot: "Hello! How can I help you today?"
- Intent: `GREETING`

**Turn 2**:
- User: "I have a question about my recent order"
- Bot: "I'd be happy to help. Could you please provide your order number?"
- Intent: `REQUEST`

**Turn 3**:
- User: "It's ORD12345"
- Bot: "Your order #ORD12345 has been shipped..."
- Intent: `REQUEST`, Tool: `get_order_status`

## System Prompts

### English Prompt

```
You are a helpful, professional customer service AI assistant for a UAE-based company.

Guidelines:
- Be polite, empathetic, and professional
- Respond in the same language as the customer (Arabic or English)
- Use available tools to find information or take actions
- If you cannot help, escalate to a human agent
- Keep responses concise (2-3 sentences max)
- Never make up information - use search_kb tool to find accurate answers
- Always confirm customer's identity when accessing sensitive information

Current context:
- Time zone: Asia/Dubai (UTC+4)
- Support hours: Sunday-Thursday, 9 AM - 6 PM
```

### Arabic Prompt

```
أنت مساعد ذكاء اصطناعي محترف وودود لخدمة العملاء في شركة مقرها الإمارات.

الإرشادات:
- كن مهذباً ومتعاطفاً ومحترفاً
- استجب بنفس لغة العميل (العربية أو الإنجليزية)
- استخدم الأدوات المتاحة للعثور على المعلومات أو اتخاذ الإجراءات
- إذا لم تتمكن من المساعدة، قم بالتصعيد إلى موظف بشري
- اجعل الردود موجزة (2-3 جمل كحد أقصى)
- لا تختلق معلومات - استخدم أداة البحث للعثور على إجابات دقيقة
```

## Policy Engine

### Automatic Escalation Rules

The service includes a policy engine that automatically escalates conversations based on:

1. **Emotion Detection**: Angry or frustrated emotions
2. **Escalation Keywords**: "speak to human", "manager", "موظف بشري", "مدير"
3. **Conversation Length**: More than 10 turns (indicates stuck loop)
4. **Explicit Requests**: User explicitly asks for human agent
5. **Tool Failures**: Multiple failed tool executions

**Example**:
```python
should_escalate, reason = policy_engine.should_escalate(
    user_text="This is terrible, I want to speak to a manager!",
    emotion="angry",
    turn_count=3,
    language="en"
)
# Returns: (True, "Detected angry emotion")
```

## LLM Configuration

### vLLM Server Setup

The NLU service requires a running vLLM server with OpenAI-compatible API.

**Recommended Models**:
- **Qwen/Qwen2.5-72B-Instruct** (Best quality)
- **meta-llama/Llama-3.1-70B-Instruct** (Alternative)
- **Qwen/Qwen2.5-32B-Instruct** (Lower resource usage)

**vLLM Configuration**:
```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --trust-remote-code
```

**GPU Requirements**:
- 72B model: 2x A100 (80GB) or 2x H100
- 32B model: 1x A100 (40GB) or 1x A10 (24GB with quantization)

### Model Parameters

```python
{
  "temperature": 0.7,        # Randomness (0=deterministic, 1=creative)
  "max_tokens": 200,         # Max response length
  "tool_choice": "auto",     # Automatic tool calling
  "stream": False            # Non-streaming (use streaming for real-time)
}
```

## Observability

### Langfuse Integration

The service logs all LLM interactions to Langfuse for monitoring and debugging.

**Tracked Metrics**:
- User input and bot response
- Tool calls executed
- Inference time
- Model used
- Session/call ID

**View in Langfuse**:
```
https://cloud.langfuse.com/projects/{project}/traces
```

**Example Trace**:
```json
{
  "name": "nlu_turn",
  "session_id": "call_a1b2c3d4",
  "input": "What are your return policies?",
  "output": "Items can be returned within 30 days...",
  "metadata": {
    "tools_called": ["search_kb"],
    "inference_time_ms": 342.5,
    "model": "Qwen/Qwen2.5-72B-Instruct"
  }
}
```

### OpenTelemetry Traces

Distributed traces are exported to OpenTelemetry collector.

**Spans**:
- `nlu_process` - Full turn processing
- `vllm_call` - LLM inference
- `tool_execution` - Individual tool calls

## Performance

### Benchmarks

**Hardware**: 2x A100 (80GB)

| Operation | Latency (p50) | Latency (p95) | Notes |
|-----------|---------------|---------------|-------|
| Intent classification | 150-200ms | 300ms | Without tool calls |
| With tool call (RAG) | 400-500ms | 800ms | Includes RAG query |
| With tool call (CRM) | 300-400ms | 600ms | Mock CRM data |
| Multi-tool execution | 600-800ms | 1200ms | 2-3 tools |

### Optimization Tips

1. **Reduce Max Tokens**: Lower `max_tokens` for faster responses
2. **Use Smaller Model**: Qwen2.5-32B instead of 72B
3. **Parallel Tool Execution**: Execute independent tools concurrently
4. **Cache Common Queries**: Cache frequent RAG queries
5. **Streaming**: Use streaming for real-time responsiveness

## Error Handling

### vLLM Server Unavailable

```python
# Automatic retry with exponential backoff
# Falls back to generic response if vLLM fails
```

### Tool Execution Failures

```python
# Tool errors are logged but don't crash the turn
# LLM generates response acknowledging the issue
```

**Example**:
```
User: "Check my order #ORD12345"
Tool: get_order_status → Error: Order not found
Response: "I'm sorry, I couldn't find order #ORD12345. Could you please verify the order number?"
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest test_nlu.py -v

# With coverage
pytest test_nlu.py --cov=policy --cov-report=html
```

### Code Structure

```
nlu_service/
├── policy.py           # Main NLU service implementation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
└── README.md           # This file
```

### Adding New Tools

1. **Define tool schema** in `TOOLS` array:
```python
{
    "type": "function",
    "function": {
        "name": "check_warranty",
        "description": "Check warranty status for a product",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"}
            },
            "required": ["product_id"]
        }
    }
}
```

2. **Implement tool execution**:
```python
async def _check_warranty(self, product_id: str, tenant_id: str) -> Dict:
    # Call warranty service
    return {"product_id": product_id, "warranty_valid": True}
```

3. **Add to `_execute_tool` method**:
```python
elif tool == ActionType.CHECK_WARRANTY:
    return await self._check_warranty(params["product_id"], tenant_id)
```

## Troubleshooting

### vLLM Connection Error

**Problem**: `Connection refused` when calling vLLM

**Solution**:
```bash
# Check vLLM is running
curl http://localhost:8000/v1/models

# Verify vLLM logs
docker logs vllm-server

# Check firewall/network
telnet localhost 8000
```

---

### Slow Response Times

**Problem**: NLU responses taking >2 seconds

**Solutions**:
1. Use smaller model (Qwen2.5-32B)
2. Reduce `max_tokens` to 150
3. Enable GPU if running on CPU
4. Check vLLM GPU utilization: `nvidia-smi`

---

### Tool Execution Failures

**Problem**: Tools returning errors

**Solution**:
```bash
# Check RAG service is running
curl http://localhost:8080/health

# Check CRM connector
curl http://localhost:8090/health

# Review logs for specific error
docker logs nlu-service
```

---

### Langfuse Not Logging

**Problem**: Traces not appearing in Langfuse

**Solution**:
```bash
# Verify Langfuse credentials
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."

# Check network connectivity
curl https://cloud.langfuse.com/api/public/health

# Review logs for auth errors
docker logs nlu-service | grep langfuse
```

## Integration Examples

### Python Client

```python
import httpx
import asyncio

async def chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/process",
            json={
                "call_id": "test_call_001",
                "text": "What are your support hours?",
                "tenant_id": "tenant_123",
                "language": "en"
            }
        )
        result = response.json()
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['text_response']}")
        print(f"Tools called: {len(result['tool_calls'])}")

asyncio.run(chat())
```

### cURL

```bash
# Simple query
curl -X POST http://localhost:8001/process \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "test_001",
    "text": "Check my order ORD12345",
    "tenant_id": "tenant_123",
    "language": "en"
  }'

# Arabic query
curl -X POST http://localhost:8001/process \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "test_002",
    "text": "ما هي ساعات الدعم؟",
    "tenant_id": "tenant_123",
    "language": "ar"
  }'
```

## Best Practices

1. **Always provide context**: Include `user_id`, `customer_id` in context
2. **Clear history regularly**: Clear conversation history after call ends
3. **Monitor token usage**: Track `inference_time_ms` for performance
4. **Handle code-switching**: Set `language="auto"` for mixed Arabic/English
5. **Test escalation logic**: Ensure escalation triggers work correctly
6. **Cache embeddings**: Cache common queries for faster RAG responses
7. **Use streaming**: Enable streaming for real-time user experience
8. **Validate tool results**: Check tool outputs before generating final response

## Security Considerations

- **PII Protection**: Redact sensitive data before logging to Langfuse
- **Input Validation**: Validate all user inputs and tool parameters
- **Rate Limiting**: Implement per-tenant rate limits
- **Tool Authorization**: Verify tenant has access to requested tools
- **Prompt Injection Defense**: Filter malicious prompts attempting to override system instructions

## Roadmap

- [ ] Streaming response generation
- [ ] Multi-language tool descriptions
- [ ] Automatic prompt optimization
- [ ] A/B testing for different prompts
- [ ] Custom tool definitions per tenant
- [ ] Fine-tuned intent classifier
- [ ] Sentiment analysis integration
- [ ] Voice tone recommendations for TTS

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- Architecture: [Architecture Docs](../docs/architecture.md)
