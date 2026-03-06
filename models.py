"""
TuneVid.com — SQLAlchemy ORM Models
"""
import uuid
from datetime import datetime, date

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    Text,
    Date,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, nullable=False, index=True)
    name = Column(String(255))
    avatar = Column(Text)
    password_hash = Column(String(255))
    plan_type = Column(
        String(20), nullable=False, default="free",
    )
    subscription_end_date = Column(DateTime(timezone=True))
    referral_code = Column(String(12), unique=True, index=True)
    google_sub = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    usage_limits = relationship("UsageLimit", back_populates="user", cascade="all, delete-orphan")
    uploads_monthly = relationship("YouTubeUploadMonthly", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    referrals_given = relationship("Referral", foreign_keys="Referral.referrer_id", back_populates="referrer")
    referrals_received = relationship("Referral", foreign_keys="Referral.referee_id", back_populates="referee")

    __table_args__ = (
        CheckConstraint("plan_type IN ('free', 'pro', 'max')", name="ck_users_plan"),
    )


class Device(Base):
    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    device_fingerprint_hash = Column(String(128), nullable=False, index=True)
    ip_hash = Column(String(128), nullable=False, index=True)
    user_agent = Column(Text)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="devices")

    __table_args__ = (
        UniqueConstraint("user_id", "device_fingerprint_hash", name="uq_device_user_fp"),
    )


class UsageLimit(Base):
    __tablename__ = "usage_limits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id", ondelete="SET NULL"))
    tool_name = Column(String(64), nullable=False)
    usage_count_24h = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), server_default=func.now())
    window_start = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="usage_limits")

    __table_args__ = (
        UniqueConstraint("user_id", "tool_name", name="uq_usage_user_tool"),
        Index("idx_usage_user_tool", "user_id", "tool_name"),
    )


class YouTubeUploadMonthly(Base):
    __tablename__ = "youtube_uploads_monthly"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id", ondelete="SET NULL"))
    count = Column(Integer, nullable=False, default=0)
    period_start = Column(Date, nullable=False, default=lambda: date.today().replace(day=1))

    user = relationship("User", back_populates="uploads_monthly")

    __table_args__ = (
        UniqueConstraint("user_id", "period_start", name="uq_yt_uploads_user_period"),
        Index("idx_yt_uploads_user", "user_id", "period_start"),
    )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    razorpay_subscription_id = Column(String(255))
    razorpay_order_id = Column(String(255), index=True)
    razorpay_payment_id = Column(String(255))
    plan_type = Column(String(20), nullable=False)
    amount_paise = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="INR")
    status = Column(String(30), nullable=False, default="created")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="subscriptions")

    __table_args__ = (
        CheckConstraint("plan_type IN ('pro', 'max')", name="ck_subs_plan"),
        CheckConstraint(
            "status IN ('created', 'paid', 'failed', 'refunded', 'expired')",
            name="ck_subs_status",
        ),
    )


class Referral(Base):
    __tablename__ = "referrals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    referrer_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    referee_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(30), nullable=False, default="pending")
    reward_applied = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    referrer = relationship("User", foreign_keys=[referrer_id], back_populates="referrals_given")
    referee = relationship("User", foreign_keys=[referee_id], back_populates="referrals_received")

    __table_args__ = (
        UniqueConstraint("referrer_id", "referee_id", name="uq_referral_pair"),
        CheckConstraint("status IN ('pending', 'qualified', 'rewarded')", name="ck_referral_status"),
    )


class BillingEvent(Base):
    __tablename__ = "billing_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    razorpay_event_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSONB, nullable=False)
    processed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
