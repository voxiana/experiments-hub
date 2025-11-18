# Connectors Service

## Overview

The Connectors Service provides unified integrations with CRM systems (Salesforce, Zendesk, HubSpot, Freshdesk) and live chat platforms (LiveChat) for the Voice AI CX Platform. It enables the NLU service to access customer data, create support tickets, and facilitate seamless handoffs to human agents.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Connectors Service                        │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │    CRM     │  │  LiveChat   │  │   Unified    │      │
│  │ Connectors │  │  Connector  │  │     API      │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│         ┌─────────────────┴──────────────────┐            │
│         │                                    │             │
│    ┌────▼────┐  ┌──────────┐  ┌──────────┐ │            │
│    │Salesforce│ │ Zendesk  │  │ HubSpot  │ │            │
│    │   API    │  │   API    │  │   API    │ │            │
│    └──────────┘  └──────────┘  └──────────┘ │            │
│         │                                    │             │
│    ┌────▼────┐  ┌──────────┐                │            │
│    │Freshdesk│  │LiveChat  │                │            │
│    │   API    │  │   API    │                │            │
│    └──────────┘  └──────────┘                │            │
└──────────────────────────────────────────────────────────┘
```

## Features

### CRM Integrations

- **Salesforce**: Cases, Contacts, Accounts
- **Zendesk**: Tickets, Users
- **HubSpot**: Contacts, Deals, Tickets
- **Freshdesk**: Tickets, Contacts

### Live Chat Integration

- **LiveChat**: Real-time agent handoff with context transfer

### Core Capabilities

- **Unified Interface**: Single API for multiple CRM systems
- **Customer Lookup**: Retrieve customer information
- **Ticket Management**: Create, update, and retrieve tickets
- **Human Handoff**: Transfer conversations to live agents
- **Context Preservation**: Pass conversation history and metadata
- **OAuth Support**: Secure authentication
- **Error Handling**: Graceful fallbacks and retries

## Technology Stack

### Core Framework

- **Python 3.10+** - Runtime environment
- **FastAPI 0.104.1** - API framework
- **aiohttp 3.9.0** - Async HTTP client
- **Pydantic 2.5.0** - Data validation

### Authentication

- **OAuth 2.0** - For CRM platforms
- **API Keys** - For simplified authentication

## Configuration

### Environment Variables

```bash
# Server
HOST="0.0.0.0"
PORT=8090

# Salesforce
SALESFORCE_CLIENT_ID="your_client_id"
SALESFORCE_CLIENT_SECRET="your_client_secret"
SALESFORCE_INSTANCE_URL="https://your-instance.salesforce.com"

# Zendesk
ZENDESK_SUBDOMAIN="your-company"
ZENDESK_EMAIL="admin@yourcompany.com"
ZENDESK_API_TOKEN="your_api_token"

# HubSpot
HUBSPOT_API_KEY="your_api_key"

# Freshdesk
FRESHDESK_DOMAIN="yourcompany.freshdesk.com"
FRESHDESK_API_KEY="your_api_key"

# LiveChat
LIVECHAT_LICENSE_ID="your_license_id"
LIVECHAT_API_KEY="your_api_key"
```

## Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure CRM credentials** (see Configuration section)

3. **Run the service**:
```bash
python server.py
# Service will start on http://0.0.0.0:8090
```

### Docker Deployment

```bash
docker build -t connectors:latest .
docker run -p 8090:8090 \
  -e ZENDESK_SUBDOMAIN="your-company" \
  -e ZENDESK_API_TOKEN="your_token" \
  connectors:latest
```

## API Reference

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy"
}
```

---

### Get Customer

Retrieve customer information from CRM.

**Endpoint**: `POST /crm/customer`

**Request Body**:
```json
{
  "crm_type": "salesforce",
  "customer_id": "003XXXXXXXXXXXXXX",
  "config": {
    "access_token": "your_token"
  }
}
```

**Parameters**:
- `crm_type`: CRM platform (`salesforce`, `zendesk`, `hubspot`, `freshdesk`)
- `customer_id`: CRM-specific customer/contact ID
- `config`: Platform-specific configuration (credentials)

**Response**:
```json
{
  "id": "003XXXXXXXXXXXXXX",
  "name": "Ahmed Al-Mansoori",
  "email": "ahmed@example.ae",
  "phone": "+971-50-123-4567",
  "tier": "Enterprise",
  "metadata": {
    "account_type": "business",
    "created_date": "2023-01-15"
  }
}
```

---

### Get Ticket

Retrieve ticket/case details from CRM.

**Endpoint**: `POST /crm/ticket`

**Request Body**:
```json
{
  "crm_type": "zendesk",
  "ticket_id": "12345",
  "config": {
    "subdomain": "your-company",
    "email": "admin@yourcompany.com",
    "api_token": "your_token"
  }
}
```

**Response**:
```json
{
  "id": "12345",
  "subject": "Product not working",
  "description": "Customer reports issue with...",
  "status": "open",
  "priority": "high",
  "customer_id": "user_123",
  "created_at": "2025-11-18T10:30:00Z",
  "updated_at": "2025-11-18T11:45:00Z",
  "metadata": {
    "tags": ["technical", "urgent"]
  }
}
```

---

### Create Handoff

Create a live chat handoff to human agent.

**Endpoint**: `POST /handoff/create`

**Request Body**:
```json
{
  "chat_type": "livechat",
  "context": {
    "call_id": "call_abc123",
    "customer_name": "Ahmed Al-Mansoori",
    "customer_email": "ahmed@example.ae",
    "transcript": [
      {"role": "user", "text": "I need help with my order"},
      {"role": "bot", "text": "I'd be happy to help. What's your order number?"},
      {"role": "user", "text": "ORD12345"}
    ],
    "reason": "customer_request",
    "urgency": "medium",
    "metadata": {
      "order_id": "ORD12345",
      "language": "en"
    }
  },
  "config": {
    "license_id": "your_license_id",
    "api_key": "your_api_key"
  }
}
```

**Response**:
```json
{
  "handoff_id": "handoff_xyz789",
  "chat_url": "https://livechat.com/agent/chat/xyz789",
  "agent_name": "Sara Ahmed",
  "agent_id": "agent_456",
  "estimated_wait_seconds": 30,
  "status": "queued"
}
```

## Data Models

### Customer

```python
class Customer(BaseModel):
    id: str                     # CRM customer ID
    name: str                   # Full name
    email: Optional[str]        # Email address
    phone: Optional[str]        # Phone number
    tier: Optional[str]         # Customer tier (Gold, Enterprise, etc.)
    metadata: Dict              # Additional custom fields
```

### Ticket

```python
class Ticket(BaseModel):
    id: str                     # Ticket/case ID
    subject: str                # Ticket subject/title
    description: str            # Detailed description
    status: TicketStatus        # new, open, pending, solved, closed
    priority: TicketPriority    # low, medium, high, urgent
    customer_id: Optional[str]  # Associated customer ID
    created_at: Optional[str]   # ISO 8601 timestamp
    updated_at: Optional[str]   # ISO 8601 timestamp
    metadata: Dict              # Additional custom fields
```

### HandoffContext

```python
class HandoffContext(BaseModel):
    call_id: str                # Unique call identifier
    customer_name: str          # Customer name
    customer_email: Optional[str]
    transcript: List[Dict]      # Conversation history
    reason: str                 # Handoff reason
    urgency: str                # low, medium, high
    metadata: Dict              # Additional context
```

## Supported Platforms

### Salesforce

**Authentication**: OAuth 2.0

**Required Permissions**:
- Read Contacts
- Read/Write Cases
- Read Accounts

**Configuration**:
```python
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "instance_url": "https://your-instance.salesforce.com",
  "access_token": "session_token"  # Obtained via OAuth flow
}
```

**API Endpoints Used**:
- `/services/data/v57.0/sobjects/Contact/{id}`
- `/services/data/v57.0/sobjects/Case`

---

### Zendesk

**Authentication**: Basic Auth (Email + API Token)

**Configuration**:
```python
{
  "subdomain": "your-company",
  "email": "admin@yourcompany.com",
  "api_token": "your_api_token"
}
```

**API Endpoints Used**:
- `/api/v2/users/{id}.json`
- `/api/v2/tickets.json`
- `/api/v2/tickets/{id}.json`

---

### HubSpot

**Authentication**: API Key

**Configuration**:
```python
{
  "api_key": "your_api_key"
}
```

**API Endpoints Used**:
- `/crm/v3/objects/contacts/{id}`
- `/crm/v3/objects/tickets`

---

### Freshdesk

**Authentication**: API Key

**Configuration**:
```python
{
  "domain": "yourcompany.freshdesk.com",
  "api_key": "your_api_key"
}
```

**API Endpoints Used**:
- `/api/v2/contacts/{id}`
- `/api/v2/tickets`

---

### LiveChat

**Authentication**: API Key + License ID

**Configuration**:
```python
{
  "license_id": "your_license_id",
  "api_key": "your_api_key"
}
```

**API Endpoints Used**:
- `/v3/agent/chats`
- `/v3/agent/handoffs`

## Usage Examples

### Python Client

```python
import requests

# Get customer from Salesforce
response = requests.post(
    "http://localhost:8090/crm/customer",
    json={
        "crm_type": "salesforce",
        "customer_id": "003XXXXXXXXXXXXXX",
        "config": {
            "access_token": "your_token"
        }
    }
)
customer = response.json()
print(f"Customer: {customer['name']}")

# Get ticket from Zendesk
response = requests.post(
    "http://localhost:8090/crm/ticket",
    json={
        "crm_type": "zendesk",
        "ticket_id": "12345",
        "config": {
            "subdomain": "your-company",
            "email": "admin@yourcompany.com",
            "api_token": "your_token"
        }
    }
)
ticket = response.json()
print(f"Ticket: {ticket['subject']} - {ticket['status']}")

# Create LiveChat handoff
response = requests.post(
    "http://localhost:8090/handoff/create",
    json={
        "chat_type": "livechat",
        "context": {
            "call_id": "call_123",
            "customer_name": "Ahmed",
            "customer_email": "ahmed@example.ae",
            "transcript": [
                {"role": "user", "text": "I need help"}
            ],
            "reason": "customer_request",
            "urgency": "medium",
            "metadata": {}
        },
        "config": {
            "license_id": "your_license",
            "api_key": "your_key"
        }
    }
)
handoff = response.json()
print(f"Handoff created: {handoff['chat_url']}")
```

### cURL Examples

```bash
# Get customer from HubSpot
curl -X POST http://localhost:8090/crm/customer \
  -H "Content-Type: application/json" \
  -d '{
    "crm_type": "hubspot",
    "customer_id": "12345",
    "config": {
      "api_key": "your_api_key"
    }
  }'

# Create Freshdesk ticket
curl -X POST http://localhost:8090/crm/ticket \
  -H "Content-Type: application/json" \
  -d '{
    "crm_type": "freshdesk",
    "ticket_id": "54321",
    "config": {
      "domain": "yourcompany.freshdesk.com",
      "api_key": "your_api_key"
    }
  }'
```

## Development

### Adding New CRM Connector

1. **Create connector class** in `crm.py`:

```python
class NewCRMConnector(BaseCRMConnector):
    async def get_customer(self, customer_id: str) -> Optional[Customer]:
        # Implement customer retrieval
        pass

    async def create_ticket(self, subject: str, description: str, ...) -> Ticket:
        # Implement ticket creation
        pass

    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        # Implement ticket retrieval
        pass

    async def update_ticket(self, ticket_id: str, ...) -> Ticket:
        # Implement ticket update
        pass
```

2. **Register in factory**:

```python
class CRMFactory:
    @staticmethod
    def create(crm_type: str, config: Dict) -> BaseCRMConnector:
        if crm_type == "newcrm":
            return NewCRMConnector(config)
        # ...
```

3. **Add configuration** to environment variables

### Code Structure

```
connectors/
├── server.py           # FastAPI server
├── crm.py             # CRM connectors (Salesforce, Zendesk, HubSpot, Freshdesk)
├── livechat.py        # LiveChat connector
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
└── README.md          # This file
```

## Troubleshooting

### Authentication Failed

**Problem**: `401 Unauthorized` from CRM API

**Solutions**:
- Verify API credentials are correct
- Check token expiration (OAuth tokens expire)
- Ensure proper permissions/scopes
- Review CRM API rate limits

---

### Customer Not Found

**Problem**: Customer lookup returns `None`

**Solutions**:
- Verify customer ID format (Salesforce: 18-char, Zendesk: numeric)
- Check customer exists in CRM
- Ensure user has permission to access customer data

---

### Ticket Creation Failed

**Problem**: Cannot create tickets

**Solutions**:
- Verify required fields (subject, description)
- Check API quotas and rate limits
- Ensure proper ticket status/priority values
- Review CRM workflow rules (auto-assignment, etc.)

---

### Handoff Timeout

**Problem**: LiveChat handoff times out

**Solutions**:
- Check LiveChat agent availability
- Verify chat routing rules
- Review agent capacity/limits
- Ensure WebSocket connection is stable

## Performance

### Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Get Customer | 100-300ms | Depends on CRM API |
| Get Ticket | 100-300ms | Depends on CRM API |
| Create Ticket | 200-400ms | Includes validation |
| Create Handoff | 500-1000ms | Includes agent assignment |

### Optimization Tips

1. **Cache Customer Data**: Cache frequently accessed customers
2. **Batch Requests**: Use bulk APIs when available
3. **Connection Pooling**: Reuse HTTP connections
4. **Async Operations**: All CRM calls are async
5. **Retry Logic**: Implement exponential backoff for transient failures

## Security Considerations

- **Credential Storage**: Store API keys/secrets in environment variables or secrets manager
- **Token Refresh**: Implement automatic OAuth token refresh
- **API Key Rotation**: Regularly rotate API keys
- **Rate Limiting**: Respect CRM API rate limits
- **Audit Logging**: Log all CRM operations for compliance
- **Data Privacy**: Ensure GDPR/CCPA compliance when storing customer data

## Best Practices

1. **Error Handling**: Always handle CRM API errors gracefully
2. **Timeouts**: Set reasonable timeouts for CRM requests (5-10s)
3. **Retries**: Implement retry logic for transient failures
4. **Logging**: Log all CRM interactions for debugging
5. **Validation**: Validate data before sending to CRM APIs
6. **Testing**: Mock CRM APIs in tests
7. **Documentation**: Keep CRM API version docs handy

## Roadmap

- [ ] Microsoft Dynamics 365 integration
- [ ] ServiceNow connector
- [ ] Intercom live chat integration
- [ ] Webhook support for CRM events
- [ ] Bulk operations (batch ticket creation)
- [ ] Advanced filtering and search
- [ ] Custom field mapping per tenant
- [ ] CRM-to-CRM data sync

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- Salesforce API: [Developer Docs](https://developer.salesforce.com/)
- Zendesk API: [Developer Docs](https://developer.zendesk.com/)
- HubSpot API: [Developer Docs](https://developers.hubspot.com/)
- LiveChat API: [Developer Docs](https://developers.livechat.com/)
