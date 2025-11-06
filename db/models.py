"""
Database Models - SQLAlchemy ORM
Multi-tenant schema for calls, transcripts, users, and knowledge base
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

# ============================================================================
# Enums
# ============================================================================

class CallStatus(str, enum.Enum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ENDED = "ended"
    ESCALATED = "escalated"
    FAILED = "failed"

class TurnRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    HUMAN_AGENT = "human_agent"

class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# ============================================================================
# Core Models
# ============================================================================

class Tenant(Base):
    """Multi-tenancy model"""
    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(255), nullable=False)
    tier = Column(String(50), default="free")  # free, starter, pro, enterprise
    settings = Column(JSON, default={})

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    calls = relationship("Call", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tenant {self.id}: {self.name}>"


class User(Base):
    """User/customer model"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)

    email = Column(String(255), unique=True, nullable=True)
    phone = Column(String(50), nullable=True)
    name = Column(String(255), nullable=True)

    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    calls = relationship("Call", back_populates="user")

    # Indexes
    __table_args__ = (
        Index("idx_users_tenant_email", "tenant_id", "email"),
        Index("idx_users_phone", "phone"),
    )

    def __repr__(self):
        return f"<User {self.id}: {self.email or self.phone}>"


class Call(Base):
    """Call session model"""
    __tablename__ = "calls"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)

    status = Column(Enum(CallStatus), default=CallStatus.INITIALIZED, nullable=False)
    language = Column(String(10), nullable=True)  # ar, en, auto
    voice_id = Column(String(50), nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Metrics
    turn_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, nullable=True)
    escalated = Column(Boolean, default=False)
    resolved = Column(Boolean, nullable=True)

    metadata = Column(JSON, default={})

    # Relationships
    tenant = relationship("Tenant", back_populates="calls")
    user = relationship("User", back_populates="calls")
    turns = relationship("Turn", back_populates="call", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="call", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_calls_tenant_status", "tenant_id", "status"),
        Index("idx_calls_started_at", "started_at"),
        Index("idx_calls_user_id", "user_id"),
    )

    def __repr__(self):
        return f"<Call {self.id}: {self.status}>"


class Turn(Base):
    """Conversation turn (single exchange)"""
    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(36), ForeignKey("calls.id"), nullable=False)

    turn_index = Column(Integer, nullable=False)
    role = Column(Enum(TurnRole), nullable=False)
    text = Column(Text, nullable=False)

    # Audio metadata
    audio_duration_seconds = Column(Float, nullable=True)
    audio_url = Column(String(512), nullable=True)

    # ASR metadata
    language = Column(String(10), nullable=True)
    asr_confidence = Column(Float, nullable=True)

    # NLU metadata
    intent = Column(String(100), nullable=True)
    intent_confidence = Column(Float, nullable=True)

    # Sentiment/emotion
    sentiment = Column(Float, nullable=True)  # -1 to 1
    emotion = Column(String(50), nullable=True)
    emotion_scores = Column(JSON, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default={})

    # Relationships
    call = relationship("Call", back_populates="turns")

    # Indexes
    __table_args__ = (
        Index("idx_turns_call_id", "call_id"),
        Index("idx_turns_call_turn", "call_id", "turn_index"),
    )

    def __repr__(self):
        return f"<Turn {self.id}: {self.role} - {self.text[:30]}...>"


class Action(Base):
    """Tool/action execution log"""
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(36), ForeignKey("calls.id"), nullable=False)

    action_type = Column(String(100), nullable=False)  # search_kb, create_ticket, etc.
    parameters = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)

    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)

    latency_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="actions")

    # Indexes
    __table_args__ = (
        Index("idx_actions_call_id", "call_id"),
        Index("idx_actions_type", "action_type"),
    )

    def __repr__(self):
        return f"<Action {self.id}: {self.action_type}>"


class Document(Base):
    """Knowledge base document"""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)

    title = Column(String(500), nullable=False)
    source_url = Column(String(1000), nullable=True)
    source_type = Column(String(50), nullable=True)  # pdf, docx, url, text

    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    chunks_count = Column(Integer, default=0)

    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="documents")

    # Indexes
    __table_args__ = (
        Index("idx_documents_tenant_status", "tenant_id", "status"),
        Index("idx_documents_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<Document {self.id}: {self.title}>"


class Config(Base):
    """Tenant configuration (policies, thresholds, voice settings)"""
    __tablename__ = "configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)

    key = Column(String(255), nullable=False)
    value = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_configs_tenant_key", "tenant_id", "key", unique=True),
    )

    def __repr__(self):
        return f"<Config {self.tenant_id}: {self.key}>"

# ============================================================================
# Audit Log
# ============================================================================

class AuditLog(Base):
    """Immutable audit log for compliance"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(36), nullable=False)

    event_type = Column(String(100), nullable=False)  # call_started, handoff, etc.
    entity_type = Column(String(100), nullable=True)  # call, user, document
    entity_id = Column(String(36), nullable=True)

    user_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)

    data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_audit_tenant_timestamp", "tenant_id", "timestamp"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
    )

    def __repr__(self):
        return f"<AuditLog {self.id}: {self.event_type}>"
