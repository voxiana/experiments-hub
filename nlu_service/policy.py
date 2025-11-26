"""
NLU Service - Conversational AI Policy & Tool Router
Uses vLLM (Qwen2.5-72B or Llama-3.1-70B) for intent understanding and response generation
Supports tool calling, RAG integration, and multi-turn context management
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field

import aiohttp
from opentelemetry import trace
from langfuse import Langfuse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

# ============================================================================
# Configuration
# ============================================================================

# vLLM server endpoint (OpenAI-compatible API)
VLLM_URL = "http://vllm:8000/v1"
MODEL_NAME = "Qwen/Qwen3-8B"  # or "meta-llama/Llama-3.1-70B-Instruct"

# RAG service
RAG_URL = "http://rag-service:8080"

# CRM connectors
CRM_URL = "http://connectors:8090"

# Langfuse for LLM observability
langfuse = Langfuse()

# ============================================================================
# Intent & Action Schemas
# ============================================================================

class IntentType(str, Enum):
    """Conversation intents"""
    GREETING = "greeting"
    QUESTION = "question"
    COMPLAINT = "complaint"
    REQUEST = "request"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

class ActionType(str, Enum):
    """Available actions/tools"""
    SEARCH_KB = "search_kb"
    GET_ORDER_STATUS = "get_order_status"
    UPDATE_TICKET = "update_ticket"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    SEND_EMAIL = "send_email"
    SEND_SMS = "send_sms"
    GET_CUSTOMER_INFO = "get_customer_info"
    CREATE_TICKET = "create_ticket"

class ToolCall(BaseModel):
    """Tool call schema"""
    tool: ActionType
    parameters: Dict[str, Any]

class NLUResponse(BaseModel):
    """NLU service response"""
    intent: IntentType
    confidence: float
    text_response: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    emotion: Optional[str] = None
    should_escalate: bool = False
    metadata: Dict = Field(default_factory=dict)

# ============================================================================
# Tool Definitions (JSON Schema for LLM function calling)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for information about products, policies, or procedures",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["product", "policy", "technical", "billing"],
                        "description": "Category to search within",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID or number",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the conversation to a human agent when unable to help or customer explicitly requests",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for escalation",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Urgency level",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_info",
            "description": "Retrieve customer information from CRM",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID or phone number",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket in the CRM system",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Ticket subject/title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                },
                "required": ["subject", "description"],
            },
        },
    },
]

# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT_EN = """You are a helpful, professional customer service AI assistant for a UAE-based company.

Guidelines:
- Be polite, empathetic, and professional
- Respond in the same language as the customer (Arabic or English)
- For Arabic customers, use appropriate formal register and greetings
- Use available tools to find information or take actions
- If you cannot help, escalate to a human agent
- Keep responses concise (2-3 sentences max)
- Never make up information - use search_kb tool to find accurate answers
- Always confirm customer's identity when accessing sensitive information
- IMPORTANT: Execute tasks directly without showing reasoning or thinking steps. Do not explain your thought process - just call tools and respond.

Current context:
- Time zone: Asia/Dubai (UTC+4)
- Support hours: Sunday-Thursday, 9 AM - 6 PM
- Available tools: {tools}
"""

SYSTEM_PROMPT_AR = """أنت مساعد ذكاء اصطناعي محترف وودود لخدمة العملاء في شركة مقرها الإمارات.

الإرشادات:
- كن مهذباً ومتعاطفاً ومحترفاً
- استجب بنفس لغة العميل (العربية أو الإنجليزية)
- استخدم الأدوات المتاحة للعثور على المعلومات أو اتخاذ الإجراءات
- إذا لم تتمكن من المساعدة، قم بالتصعيد إلى موظف بشري
- اجعل الردود موجزة (2-3 جمل كحد أقصى)
- لا تختلق معلومات - استخدم أداة البحث للعثور على إجابات دقيقة
- تأكد دائماً من هوية العميل عند الوصول إلى معلومات حساسة
- مهم جداً: نفذ المهام مباشرة دون إظهار خطوات التفكير أو الاستدلال. لا تشرح عملية تفكيرك - فقط استدع الأدوات وأجب.

السياق الحالي:
- المنطقة الزمنية: آسيا/دبي (UTC+4)
- ساعات الدعم: الأحد-الخميس، 9 صباحاً - 6 مساءً
"""

# ============================================================================
# NLU Service
# ============================================================================

class NLUService:
    """
    NLU Service using vLLM for conversational AI
    Handles intent classification, entity extraction, and response generation
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.conversation_history: Dict[str, List[Dict]] = {}  # call_id -> messages

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("✅ NLU Service initialized")

    async def shutdown(self):
        """Cleanup"""
        if self.session:
            await self.session.close()

    async def process_turn(
        self,
        call_id: str,
        user_text: str,
        tenant_id: str,
        language: str = "en",
        context: Optional[Dict] = None,
    ) -> NLUResponse:
        """
        Process a conversation turn
        Returns intent, response text, and tool calls
        """
        with tracer.start_as_current_span("nlu_process") as span:
            span.set_attribute("call_id", call_id)
            span.set_attribute("language", language)

            # Get or create conversation history
            if call_id not in self.conversation_history:
                self.conversation_history[call_id] = []

            history = self.conversation_history[call_id]

            # Build messages
            messages = self._build_messages(user_text, language, history, context)

            # Call vLLM with tool support
            start_time = time.time()
            llm_response = await self._call_vllm(messages, tools=TOOLS)
            inference_time = time.time() - start_time

            # Parse response
            assistant_message = llm_response["choices"][0]["message"]
            response_text = assistant_message.get("content", "")
            tool_calls_raw = assistant_message.get("tool_calls", [])

            # Parse tool calls
            tool_calls = []
            for tc in tool_calls_raw:
                func = tc["function"]
                tool_calls.append(ToolCall(
                    tool=ActionType(func["name"]),
                    parameters=json.loads(func["arguments"]),
                ))

            # Execute tools
            tool_results = {}
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call, tenant_id, context)
                tool_results[tool_call.tool] = result

            # If tools were called, generate final response with results
            if tool_calls:
                # Add tool results to context and generate final response
                final_response = await self._generate_final_response(
                    messages, tool_calls, tool_results, language
                )
                response_text = final_response

            # Update history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response_text})

            # Trim history (keep last 10 turns)
            if len(history) > 20:
                history = history[-20:]
            self.conversation_history[call_id] = history

            # Detect intent
            intent = self._classify_intent(user_text, tool_calls)

            # Check escalation conditions
            should_escalate = any(tc.tool == ActionType.ESCALATE_TO_HUMAN for tc in tool_calls)

            # Log to Langfuse
            self._log_to_langfuse(
                call_id, user_text, response_text, tool_calls, inference_time
            )

            logger.info(
                f"NLU [{call_id}] intent={intent}, tools={len(tool_calls)}, "
                f"time={inference_time:.3f}s"
            )

            return NLUResponse(
                intent=intent,
                confidence=0.9,  # placeholder
                text_response=response_text,
                tool_calls=tool_calls,
                should_escalate=should_escalate,
                metadata={
                    "inference_time_ms": inference_time * 1000,
                    "tool_results": tool_results,
                    "model": MODEL_NAME,
                },
            )

    def _build_messages(
        self,
        user_text: str,
        language: str,
        history: List[Dict],
        context: Optional[Dict],
    ) -> List[Dict]:
        """Build messages array for LLM"""
        messages = []

        # System prompt
        system_prompt = SYSTEM_PROMPT_AR if language == "ar" else SYSTEM_PROMPT_EN
        tools_desc = ", ".join([t["function"]["name"] for t in TOOLS])
        system_prompt = system_prompt.format(tools=tools_desc)

        # Add context if available
        if context:
            system_prompt += f"\n\nCustomer context:\n{json.dumps(context, indent=2)}"

        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    async def _call_vllm(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Call vLLM server (OpenAI-compatible API)
        """
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,  # Lower temperature for more direct, less creative responses
            "max_tokens": 200,
            "stream": False,
        }

        async with self.session.post(
            f"{VLLM_URL}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"vLLM error: {error}")
                raise Exception(f"vLLM API error: {resp.status}")

            result = await resp.json()
            return result

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        tenant_id: str,
        context: Optional[Dict],
    ) -> Dict:
        """Execute a tool call"""
        tool = tool_call.tool
        params = tool_call.parameters

        logger.info(f"Executing tool: {tool} with params: {params}")

        try:
            if tool == ActionType.SEARCH_KB:
                return await self._search_kb(params["query"], tenant_id)

            elif tool == ActionType.GET_ORDER_STATUS:
                return await self._get_order_status(params["order_id"], tenant_id)

            elif tool == ActionType.GET_CUSTOMER_INFO:
                return await self._get_customer_info(params["customer_id"], tenant_id)

            elif tool == ActionType.CREATE_TICKET:
                return await self._create_ticket(params, tenant_id, context)

            elif tool == ActionType.ESCALATE_TO_HUMAN:
                return {"status": "escalated", "reason": params.get("reason", "unknown")}

            else:
                return {"error": "Tool not implemented"}

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {"error": str(e)}

    async def _search_kb(self, query: str, tenant_id: str) -> Dict:
        """Search knowledge base via RAG service"""
        async with self.session.post(
            f"{RAG_URL}/query",
            json={"query": query, "tenant_id": tenant_id, "top_k": 3},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                return {
                    "results": result.get("results", []),
                    "sources": [r.get("source") for r in result.get("results", [])],
                }
            else:
                return {"error": "RAG service unavailable"}

    async def _get_order_status(self, order_id: str, tenant_id: str) -> Dict:
        """Get order status from CRM"""
        # Mock implementation
        return {
            "order_id": order_id,
            "status": "shipped",
            "tracking": "DHL123456",
            "eta": "2025-11-08",
        }

    async def _get_customer_info(self, customer_id: str, tenant_id: str) -> Dict:
        """Get customer info from CRM"""
        # Mock implementation
        return {
            "customer_id": customer_id,
            "name": "Ahmed Al-Mansoori",
            "tier": "gold",
            "phone": "+971-50-123-4567",
            "email": "ahmed@example.ae",
        }

    async def _create_ticket(self, params: Dict, tenant_id: str, context: Optional[Dict]) -> Dict:
        """Create support ticket"""
        # Mock implementation
        ticket_id = f"TICKET-{int(time.time())}"
        return {
            "ticket_id": ticket_id,
            "status": "created",
            "subject": params["subject"],
            "priority": params.get("priority", "medium"),
        }

    async def _generate_final_response(
        self,
        messages: List[Dict],
        tool_calls: List[ToolCall],
        tool_results: Dict,
        language: str,
    ) -> str:
        """
        Generate final response incorporating tool results
        """
        # Add tool results to messages
        tool_messages = []
        for tool_call in tool_calls:
            result = tool_results.get(tool_call.tool, {})
            tool_messages.append({
                "role": "tool",
                "name": tool_call.tool.value,
                "content": json.dumps(result),
            })

        messages_with_tools = messages + tool_messages

        # Call LLM again to synthesize final response
        response = await self._call_vllm(messages_with_tools, tools=[])
        return response["choices"][0]["message"]["content"]

    def _classify_intent(self, text: str, tool_calls: List[ToolCall]) -> IntentType:
        """
        Classify intent based on text and tool calls
        (Simplified - in production, use LLM classification)
        """
        text_lower = text.lower()

        if any(tc.tool == ActionType.ESCALATE_TO_HUMAN for tc in tool_calls):
            return IntentType.ESCALATION

        if any(w in text_lower for w in ["hello", "hi", "مرحبا", "السلام"]):
            return IntentType.GREETING

        if any(w in text_lower for w in ["problem", "issue", "complaint", "مشكلة", "شكوى"]):
            return IntentType.COMPLAINT

        if any(w in text_lower for w in ["order", "status", "track", "طلب", "حالة"]):
            return IntentType.REQUEST

        if any(w in text_lower for w in ["bye", "goodbye", "thanks", "شكرا", "مع السلامة"]):
            return IntentType.GOODBYE

        if tool_calls:
            return IntentType.QUESTION

        return IntentType.UNKNOWN

    def _log_to_langfuse(
        self,
        call_id: str,
        user_text: str,
        response_text: str,
        tool_calls: List[ToolCall],
        inference_time: float,
    ):
        """Log conversation to Langfuse for observability"""
        try:
            trace = langfuse.trace(
                name="nlu_turn",
                session_id=call_id,
                input=user_text,
                output=response_text,
                metadata={
                    "tools_called": [tc.tool.value for tc in tool_calls],
                    "inference_time_ms": inference_time * 1000,
                    "model": MODEL_NAME,
                },
            )
        except Exception as e:
            logger.warning(f"Langfuse logging failed: {e}")

    def clear_history(self, call_id: str):
        """Clear conversation history for a call"""
        if call_id in self.conversation_history:
            del self.conversation_history[call_id]

# ============================================================================
# Policy Engine
# ============================================================================

class PolicyEngine:
    """
    Policy engine for escalation rules and guardrails
    """

    def __init__(self):
        self.escalation_keywords = {
            "en": ["speak to human", "manager", "complaint", "angry", "terrible"],
            "ar": ["موظف بشري", "مدير", "شكوى", "غاضب"],
        }

    def should_escalate(
        self,
        user_text: str,
        emotion: Optional[str],
        turn_count: int,
        language: str = "en",
    ) -> tuple[bool, str]:
        """
        Determine if conversation should escalate to human
        Returns: (should_escalate, reason)
        """
        # Check emotion
        if emotion in ["angry", "frustrated"]:
            return True, f"Detected {emotion} emotion"

        # Check keywords
        text_lower = user_text.lower()
        keywords = self.escalation_keywords.get(language, [])
        for keyword in keywords:
            if keyword in text_lower:
                return True, f"Escalation keyword: {keyword}"

        # Check turn count (stuck in loop)
        if turn_count > 10:
            return True, "Conversation too long (>10 turns)"

        return False, ""

# ============================================================================
# Main Service Runner
# ============================================================================

async def run_service():
    """Run NLU service as standalone (for testing)"""
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="NLU Service")
    nlu = NLUService()
    await nlu.initialize()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": MODEL_NAME}

    @app.post("/process")
    async def process(
        call_id: str,
        text: str,
        tenant_id: str,
        language: str = "en",
    ):
        """Process a conversation turn"""
        result = await nlu.process_turn(call_id, text, tenant_id, language)
        return result.dict()

    @app.post("/clear/{call_id}")
    async def clear_history(call_id: str):
        """Clear conversation history"""
        nlu.clear_history(call_id)
        return {"status": "cleared"}

    @app.on_event("shutdown")
    async def shutdown_event():
        await nlu.shutdown()

    config = uvicorn.Config(app, host="0.0.0.0", port=8001)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_service())
