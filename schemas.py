"""
TuneVid.com — Pydantic Schemas (Request/Response)
"""
from datetime import datetime, date
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ── User ──────────────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    email: str
    name: Optional[str] = None
    avatar: Optional[str] = None


class UserCreate(UserBase):
    google_sub: str


class UserOut(UserBase):
    id: UUID
    plan_type: str = "free"
    subscription_end_date: Optional[datetime] = None
    referral_code: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserDashboard(UserOut):
    """Extended user info for dashboard."""
    usage_limits: List["UsageLimitOut"] = []
    youtube_uploads_this_month: int = 0
    referral_count: int = 0
    referral_rewards_pending: int = 0


# ── Usage Limits ──────────────────────────────────────────────────────────────

class UsageLimitOut(BaseModel):
    tool_name: str
    usage_count_24h: int
    max_allowed: int
    last_used_at: Optional[datetime] = None
    window_start: Optional[datetime] = None
    next_reset_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── YouTube Uploads ───────────────────────────────────────────────────────────

class YouTubeUploadStatus(BaseModel):
    count: int
    max_allowed: int
    period_start: date


# ── Subscription / Payment ────────────────────────────────────────────────────

class CreateOrderRequest(BaseModel):
    plan_type: str = Field(..., pattern="^(pro|max)$")


class CreateOrderResponse(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str
    plan_type: str
    user_email: str
    user_name: Optional[str] = None


class PaymentVerifyRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class SubscriptionOut(BaseModel):
    id: UUID
    plan_type: str
    status: str
    amount_paise: int
    currency: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Referral ──────────────────────────────────────────────────────────────────

class ReferralOut(BaseModel):
    id: UUID
    referee_email: Optional[str] = None
    referee_name: Optional[str] = None
    status: str
    reward_applied: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ApplyReferralRequest(BaseModel):
    referral_code: str


# ── Device ────────────────────────────────────────────────────────────────────

class DeviceRegister(BaseModel):
    fingerprint: str
    user_agent: Optional[str] = None


# ── Plan Limits Config ────────────────────────────────────────────────────────

PLAN_LIMITS = {
    "free": {
        "youtube_uploads_monthly": 5,
        "tool_uses_24h": 3,
        "max_file_size_mb": 50,
        "quality": "1080p",
        "batch_upload": False,
        "api_access": False,
        "priority_queue": False,
    },
    "pro": {
        "youtube_uploads_monthly": -1,  # unlimited
        "tool_uses_24h": 50,
        "max_file_size_mb": 500,
        "quality": "4K",
        "batch_upload": True,
        "api_access": False,
        "priority_queue": True,
    },
    "max": {
        "youtube_uploads_monthly": -1,  # unlimited
        "tool_uses_24h": -1,  # unlimited
        "max_file_size_mb": 2048,
        "quality": "4K",
        "batch_upload": True,
        "api_access": True,
        "priority_queue": True,
    },
}

PLAN_PRICING = {
    "pro": {"INR": 75000, "USD": 900},   # in paise/cents
    "max": {"INR": 410000, "USD": 4900},
}


# ── Generic Response ──────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    success: bool = True
