"""
LiveChat Handoff Connector
Handles escalation to human agents with context transfer
Supports LiveChat.com and generic webhook-based systems
"""

import logging
import time
from typing import Dict, List, Optional
from pydantic import BaseModel
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Models
# ============================================================================

class HandoffContext(BaseModel):
    """Context data for human handoff"""
    call_id: str
    reason: str
    urgency: str = "medium"  # low, medium, high
    transcript: List[Dict]  # List of {role, text, timestamp}
    customer_info: Optional[Dict] = None
    intent_timeline: List[str] = []
    emotion_scores: List[Dict] = []
    suggested_actions: List[str] = []
    metadata: Dict = {}

class HandoffResponse(BaseModel):
    """Response from handoff request"""
    handoff_id: str
    status: str  # queued, assigned, connected
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    estimated_wait_seconds: int = 30
    queue_position: int = 1

# ============================================================================
# LiveChat.com Connector
# ============================================================================

class LiveChatConnector:
    """
    Integration with LiveChat.com
    Uses Agent API and Customer API
    """

    def __init__(self, config: Dict):
        self.license_id = config.get("license_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.access_token = config.get("access_token")
        self.base_url = "https://api.livechatinc.com/v3.5"
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        })
        logger.info("✅ LiveChat connector initialized")

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def create_handoff(self, context: HandoffContext) -> HandoffResponse:
        """
        Create a handoff request
        Transfers conversation to human agent queue
        """
        logger.info(f"Creating LiveChat handoff for call {context.call_id}")

        # Prepare chat data
        chat_data = {
            "chat": {
                "properties": {
                    "source": {"type": "voice_ai"},
                    "call_id": {"value": context.call_id},
                    "urgency": {"value": context.urgency},
                    "reason": {"value": context.reason},
                },
                "thread": {
                    "events": self._format_transcript(context.transcript),
                },
            },
            "routing": {
                "priority": self._get_priority(context.urgency),
                "skills": self._get_required_skills(context),
            },
        }

        # Stub implementation (would POST to LiveChat API)
        handoff_id = f"handoff_{int(time.time())}"

        # Mock response
        return HandoffResponse(
            handoff_id=handoff_id,
            status="queued",
            estimated_wait_seconds=self._estimate_wait_time(context.urgency),
            queue_position=self._get_queue_position(),
        )

    async def check_handoff_status(self, handoff_id: str) -> HandoffResponse:
        """Check status of handoff request"""
        logger.info(f"Checking status of handoff {handoff_id}")

        # Stub implementation
        return HandoffResponse(
            handoff_id=handoff_id,
            status="assigned",
            agent_id="agent_123",
            agent_name="Sarah Ahmed",
            estimated_wait_seconds=0,
        )

    async def transfer_call(self, call_id: str, agent_id: str) -> Dict:
        """
        Transfer active call to agent
        Updates WebRTC/SIP routing
        """
        logger.info(f"Transferring call {call_id} to agent {agent_id}")

        # TODO: Update WebRTC session routing
        # TODO: Notify agent via WebSocket
        # TODO: Update call status in database

        return {
            "status": "transferred",
            "agent_id": agent_id,
            "transfer_time": time.time(),
        }

    def _format_transcript(self, transcript: List[Dict]) -> List[Dict]:
        """Format transcript for LiveChat events"""
        events = []
        for turn in transcript:
            events.append({
                "type": "message",
                "text": turn.get("text", ""),
                "author_id": turn.get("role", "customer"),
                "created_at": turn.get("timestamp", time.time()),
            })
        return events

    def _get_priority(self, urgency: str) -> str:
        """Map urgency to LiveChat priority"""
        mapping = {
            "low": "normal",
            "medium": "normal",
            "high": "high",
            "urgent": "urgent",
        }
        return mapping.get(urgency, "normal")

    def _get_required_skills(self, context: HandoffContext) -> List[str]:
        """Determine required agent skills based on context"""
        skills = []

        # Check language
        if any("arabic" in turn.get("language", "").lower() for turn in context.transcript):
            skills.append("arabic")
        else:
            skills.append("english")

        # Check intent
        if "complaint" in context.intent_timeline:
            skills.append("complaint_handling")

        if "technical" in context.reason.lower():
            skills.append("technical_support")

        return skills

    def _estimate_wait_time(self, urgency: str) -> int:
        """Estimate wait time based on urgency and queue depth"""
        # Stub implementation
        base_wait = {
            "urgent": 10,
            "high": 20,
            "medium": 30,
            "low": 60,
        }
        return base_wait.get(urgency, 30)

    def _get_queue_position(self) -> int:
        """Get position in queue"""
        # Stub implementation
        return 1

# ============================================================================
# Generic Webhook Handoff
# ============================================================================

class WebhookHandoffConnector:
    """
    Generic webhook-based handoff
    POSTs handoff context to configurable endpoint
    """

    def __init__(self, config: Dict):
        self.webhook_url = config.get("webhook_url")
        self.auth_token = config.get("auth_token")
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session"""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self.session = aiohttp.ClientSession(headers=headers)
        logger.info("✅ Webhook handoff connector initialized")

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def create_handoff(self, context: HandoffContext) -> HandoffResponse:
        """
        POST handoff context to webhook
        """
        logger.info(f"Posting handoff to webhook: {self.webhook_url}")

        payload = {
            "event": "handoff_requested",
            "context": context.dict(),
            "timestamp": time.time(),
        }

        try:
            async with self.session.post(
                self.webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return HandoffResponse(
                        handoff_id=result.get("handoff_id", f"handoff_{int(time.time())}"),
                        status=result.get("status", "queued"),
                        agent_id=result.get("agent_id"),
                        agent_name=result.get("agent_name"),
                        estimated_wait_seconds=result.get("estimated_wait_seconds", 30),
                    )
                else:
                    logger.error(f"Webhook returned {resp.status}")
                    raise Exception(f"Webhook error: {resp.status}")

        except Exception as e:
            logger.error(f"Webhook handoff failed: {e}", exc_info=True)
            raise

# ============================================================================
# Handoff Manager
# ============================================================================

class HandoffManager:
    """
    Manages handoff lifecycle
    Coordinates between AI system and human agents
    """

    def __init__(self, connector_type: str, config: Dict):
        if connector_type == "livechat":
            self.connector = LiveChatConnector(config)
        elif connector_type == "webhook":
            self.connector = WebhookHandoffConnector(config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

        self.active_handoffs: Dict[str, HandoffContext] = {}

    async def initialize(self):
        """Initialize connector"""
        await self.connector.initialize()

    async def close(self):
        """Cleanup"""
        await self.connector.close()

    async def request_handoff(
        self,
        call_id: str,
        reason: str,
        urgency: str,
        transcript: List[Dict],
        customer_info: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> HandoffResponse:
        """
        Request human handoff
        """
        # Build context
        context = HandoffContext(
            call_id=call_id,
            reason=reason,
            urgency=urgency,
            transcript=transcript,
            customer_info=customer_info,
            metadata=metadata or {},
        )

        # Enrich context
        context = await self._enrich_context(context)

        # Store active handoff
        self.active_handoffs[call_id] = context

        # Create handoff
        response = await self.connector.create_handoff(context)

        logger.info(
            f"✅ Handoff created: {call_id} → {response.handoff_id} "
            f"(status={response.status})"
        )

        return response

    async def _enrich_context(self, context: HandoffContext) -> HandoffContext:
        """
        Enrich context with additional data
        - Intent timeline
        - Emotion analysis summary
        - Suggested actions
        """
        # Extract intents from transcript
        intents = []
        for turn in context.transcript:
            if "intent" in turn:
                intents.append(turn["intent"])
        context.intent_timeline = intents

        # Generate suggested actions
        context.suggested_actions = self._generate_suggestions(context)

        return context

    def _generate_suggestions(self, context: HandoffContext) -> List[str]:
        """Generate suggested actions for human agent"""
        suggestions = []

        if "complaint" in context.intent_timeline:
            suggestions.append("Apologize and empathize with customer")
            suggestions.append("Offer compensation if appropriate")

        if "order" in context.reason.lower():
            suggestions.append("Check order status in CRM")
            suggestions.append("Provide tracking information")

        if context.urgency in ["high", "urgent"]:
            suggestions.append("Prioritize immediate resolution")
            suggestions.append("Escalate to supervisor if needed")

        return suggestions
