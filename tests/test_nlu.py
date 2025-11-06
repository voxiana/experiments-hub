"""
Tests for NLU Service
Unit and integration tests for intent classification and tool usage
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Import NLU components
import sys
sys.path.insert(0, '../nlu_service')

from nlu_service.policy import NLUService, IntentType, ActionType, ToolCall


@pytest.fixture
async def nlu_service():
    """Create NLU service fixture"""
    service = NLUService()
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.mark.asyncio
async def test_greeting_intent():
    """Test greeting intent classification"""
    service = NLUService()
    await service.initialize()

    # Mock vLLM response
    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello! How can I help you today?",
                    "tool_calls": [],
                }
            }]
        }

        result = await service.process_turn(
            call_id="test_call_1",
            user_text="Hello",
            tenant_id="test_tenant",
            language="en",
        )

        assert result.intent == IntentType.GREETING
        assert "hello" in result.text_response.lower() or "hi" in result.text_response.lower()
        assert len(result.tool_calls) == 0

    await service.shutdown()


@pytest.mark.asyncio
async def test_search_kb_tool():
    """Test knowledge base search tool call"""
    service = NLUService()
    await service.initialize()

    # Mock vLLM to return a search tool call
    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "search_kb",
                            "arguments": '{"query": "support hours"}',
                        }
                    }],
                }
            }]
        }

        # Mock RAG service response
        with patch.object(service, '_search_kb', new=AsyncMock()) as mock_rag:
            mock_rag.return_value = {
                "results": [{
                    "text": "Our support hours are 9 AM to 6 PM.",
                    "source": "FAQ",
                }]
            }

            result = await service.process_turn(
                call_id="test_call_2",
                user_text="What are your support hours?",
                tenant_id="test_tenant",
                language="en",
            )

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].tool == ActionType.SEARCH_KB
            assert "support hours" in result.tool_calls[0].parameters["query"].lower()

    await service.shutdown()


@pytest.mark.asyncio
async def test_escalation_intent():
    """Test escalation to human detection"""
    service = NLUService()
    await service.initialize()

    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "I'll connect you with a human agent right away.",
                    "tool_calls": [{
                        "function": {
                            "name": "escalate_to_human",
                            "arguments": '{"reason": "customer requested", "urgency": "medium"}',
                        }
                    }],
                }
            }]
        }

        result = await service.process_turn(
            call_id="test_call_3",
            user_text="I want to speak to a human",
            tenant_id="test_tenant",
            language="en",
        )

        assert result.intent == IntentType.ESCALATION
        assert result.should_escalate == True
        assert any(tc.tool == ActionType.ESCALATE_TO_HUMAN for tc in result.tool_calls)

    await service.shutdown()


@pytest.mark.asyncio
async def test_conversation_context():
    """Test multi-turn conversation context"""
    service = NLUService()
    await service.initialize()

    call_id = "test_call_4"

    # Turn 1: Ask about order
    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "I can help you check your order. What's your order number?",
                    "tool_calls": [],
                }
            }]
        }

        result1 = await service.process_turn(
            call_id=call_id,
            user_text="I want to check my order",
            tenant_id="test_tenant",
            language="en",
        )

        assert len(service.conversation_history[call_id]) == 2  # User + Assistant

    # Turn 2: Provide order number (should use context)
    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_order_status",
                            "arguments": '{"order_id": "12345"}',
                        }
                    }],
                }
            }]
        }

        with patch.object(service, '_get_order_status', new=AsyncMock()) as mock_order:
            mock_order.return_value = {"status": "shipped"}

            result2 = await service.process_turn(
                call_id=call_id,
                user_text="12345",
                tenant_id="test_tenant",
                language="en",
            )

            # Check that context was maintained
            assert len(service.conversation_history[call_id]) == 4  # 2 previous + 2 new

    await service.shutdown()


def test_intent_classification_helper():
    """Test intent classification helper function"""
    service = NLUService()

    # Greeting
    intent = service._classify_intent("Hello", [])
    assert intent == IntentType.GREETING

    # Question with tool
    tool_call = ToolCall(tool=ActionType.SEARCH_KB, parameters={"query": "test"})
    intent = service._classify_intent("What are your hours?", [tool_call])
    assert intent == IntentType.QUESTION

    # Escalation
    tool_call = ToolCall(tool=ActionType.ESCALATE_TO_HUMAN, parameters={"reason": "test"})
    intent = service._classify_intent("I want a human", [tool_call])
    assert intent == IntentType.ESCALATION


@pytest.mark.asyncio
async def test_arabic_support():
    """Test Arabic language support"""
    service = NLUService()
    await service.initialize()

    with patch.object(service, '_call_vllm', new=AsyncMock()) as mock_vllm:
        mock_vllm.return_value = {
            "choices": [{
                "message": {
                    "content": "مرحباً! كيف يمكنني مساعدتك؟",
                    "tool_calls": [],
                }
            }]
        }

        result = await service.process_turn(
            call_id="test_call_ar",
            user_text="مرحبا",
            tenant_id="test_tenant",
            language="ar",
        )

        assert result.intent == IntentType.GREETING

    await service.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
