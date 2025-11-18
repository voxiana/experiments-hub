"""
CRM Connectors Service - FastAPI Server
Provides unified API for CRM and LiveChat integrations
"""

import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from crm import CRMFactory, BaseCRMConnector
from livechat import LiveChatFactory, HandoffContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CRM Connectors Service", version="1.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class GetCustomerRequest(BaseModel):
    crm_type: str
    customer_id: str
    config: Dict

class GetTicketRequest(BaseModel):
    crm_type: str
    ticket_id: str
    config: Dict

class CreateHandoffRequest(BaseModel):
    chat_type: str
    context: Dict
    config: Dict

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.post("/crm/customer")
async def get_customer(request: GetCustomerRequest):
    """Get customer from CRM"""
    try:
        connector = CRMFactory.create(request.crm_type, request.config)
        customer = await connector.get_customer(request.customer_id)
        return customer.dict() if customer else None
    except Exception as e:
        logger.error(f"Error getting customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crm/ticket")
async def get_ticket(request: GetTicketRequest):
    """Get ticket from CRM"""
    try:
        connector = CRMFactory.create(request.crm_type, request.config)
        ticket = await connector.get_ticket(request.ticket_id)
        return ticket.dict() if ticket else None
    except Exception as e:
        logger.error(f"Error getting ticket: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handoff/create")
async def create_handoff(request: CreateHandoffRequest):
    """Create live chat handoff"""
    try:
        factory = LiveChatFactory()
        connector = factory.create(request.chat_type, request.config)
        
        context = HandoffContext(**request.context)
        response = await connector.create_handoff(context)
        return response.dict()
    except Exception as e:
        logger.error(f"Error creating handoff: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8090,
        log_level="info",
    )

