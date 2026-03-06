"""
TuneVid.com — Dashboard & User API
Provides user profile, usage statistics, and dashboard data.
"""
import logging
from datetime import datetime, timedelta, date

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, UsageLimit, YouTubeUploadMonthly, Referral
from schemas import PLAN_LIMITS, UserOut

logger = logging.getLogger("tunevid.dashboard")

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/me")
async def get_current_user_profile(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user profile with plan info."""
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "avatar": user.avatar,
        "plan_type": user.plan_type,
        "subscription_end_date": user.subscription_end_date.isoformat() if user.subscription_end_date else None,
        "referral_code": user.referral_code,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "plan_limits": PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"]),
    }


@router.get("/dashboard")
async def get_dashboard_data(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get complete dashboard data including usage stats, limits, and referral info.
    """
    plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
    now = datetime.utcnow()

    # ── Usage Limits (per tool) ──
    stmt = select(UsageLimit).where(UsageLimit.user_id == user.id)
    result = await db.execute(stmt)
    usage_records = result.scalars().all()

    tool_usage = []
    for usage in usage_records:
        # Check if 24h window has reset
        window_elapsed = now - usage.window_start
        current_count = usage.usage_count_24h
        if window_elapsed >= timedelta(hours=24):
            current_count = 0  # Would reset on next use

        max_allowed = plan["tool_uses_24h"]
        reset_at = (usage.window_start + timedelta(hours=24)).isoformat() if max_allowed != -1 else None

        tool_usage.append({
            "tool_name": usage.tool_name,
            "used": current_count,
            "max": max_allowed,
            "remaining": max(0, max_allowed - current_count) if max_allowed != -1 else -1,
            "last_used_at": usage.last_used_at.isoformat() if usage.last_used_at else None,
            "reset_at": reset_at,
        })

    # ── YouTube Uploads This Month ──
    today = now.date()
    period_start = today.replace(day=1)

    yt_stmt = select(YouTubeUploadMonthly).where(
        YouTubeUploadMonthly.user_id == user.id,
        YouTubeUploadMonthly.period_start == period_start,
    )
    yt_result = await db.execute(yt_stmt)
    yt_monthly = yt_result.scalar_one_or_none()

    yt_used = yt_monthly.count if yt_monthly else 0
    yt_max = plan["youtube_uploads_monthly"]

    # ── Referral Stats ──
    referral_counts_stmt = (
        select(Referral.status, func.count())
        .where(Referral.referrer_id == user.id)
        .group_by(Referral.status)
    )
    referral_counts_result = await db.execute(referral_counts_stmt)
    referral_counts = {status: count for status, count in referral_counts_result.all()}
    referral_total = sum(referral_counts.values())
    referral_rewarded = referral_counts.get("rewarded", 0)
    referral_pending = referral_counts.get("pending", 0)

    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "avatar": user.avatar,
            "plan_type": user.plan_type,
            "subscription_end_date": user.subscription_end_date.isoformat() if user.subscription_end_date else None,
            "referral_code": user.referral_code,
        },
        "plan_limits": plan,
        "tool_usage": tool_usage,
        "youtube_uploads": {
            "used": yt_used,
            "max": yt_max,
            "remaining": max(0, yt_max - yt_used) if yt_max != -1 else -1,
            "period_start": period_start.isoformat(),
        },
        "referrals": {
            "total": referral_total,
            "rewarded": referral_rewarded,
            "pending": referral_pending,
            "referral_code": user.referral_code,
            "referral_link": f"https://tunevid.com/?ref={user.referral_code}",
        },
    }


@router.get("/tool-usage/{tool_name}")
async def get_tool_usage(
    tool_name: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get usage info for a specific tool. Used by frontend to show
    remaining uses and countdown timer.
    """
    plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
    max_uses = plan["tool_uses_24h"]
    now = datetime.utcnow()

    stmt = select(UsageLimit).where(
        UsageLimit.user_id == user.id,
        UsageLimit.tool_name == tool_name,
    )
    result = await db.execute(stmt)
    usage = result.scalar_one_or_none()

    if not usage:
        return {
            "tool_name": tool_name,
            "used": 0,
            "max": max_uses,
            "remaining": max_uses if max_uses != -1 else -1,
            "reset_at": None,
        }

    # Check if 24h window has reset
    window_elapsed = now - usage.window_start
    current_count = usage.usage_count_24h
    if window_elapsed >= timedelta(hours=24):
        current_count = 0

    reset_at = (usage.window_start + timedelta(hours=24)).isoformat() if max_uses != -1 and current_count > 0 else None

    return {
        "tool_name": tool_name,
        "used": current_count,
        "max": max_uses,
        "remaining": max(0, max_uses - current_count) if max_uses != -1 else -1,
        "reset_at": reset_at,
    }


@router.get("/youtube-upload-status")
async def get_youtube_upload_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get YouTube upload status — daily limit for free users (1/day).
    Returns remaining uploads and countdown to next reset.
    """
    plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
    now = datetime.utcnow()

    # Check daily YouTube upload limit
    stmt = select(UsageLimit).where(
        UsageLimit.user_id == user.id,
        UsageLimit.tool_name == "youtube_upload",
    )
    result = await db.execute(stmt)
    usage = result.scalar_one_or_none()

    daily_max = 1 if user.plan_type == "free" else (-1 if user.plan_type in ("pro", "max") else 1)

    if not usage:
        return {
            "used_today": 0,
            "max_daily": daily_max,
            "remaining": daily_max if daily_max != -1 else -1,
            "can_upload": True,
            "reset_at": None,
        }

    window_elapsed = now - usage.window_start
    current_count = usage.usage_count_24h
    if window_elapsed >= timedelta(hours=24):
        current_count = 0

    can_upload = daily_max == -1 or current_count < daily_max
    reset_at = (usage.window_start + timedelta(hours=24)).isoformat() if not can_upload else None

    return {
        "used_today": current_count,
        "max_daily": daily_max,
        "remaining": max(0, daily_max - current_count) if daily_max != -1 else -1,
        "can_upload": can_upload,
        "reset_at": reset_at,
    }
