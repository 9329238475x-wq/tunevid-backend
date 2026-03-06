"""
TuneVid.com — Security Middleware & Usage Limit Dependencies
Controls device fingerprinting, IP abuse prevention, and per-tool usage limits.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user, register_device, _hash_value
from database import get_db
from models import User, Device, UsageLimit, YouTubeUploadMonthly
from schemas import PLAN_LIMITS

logger = logging.getLogger("tunevid.middleware")


# ── USAGE LIMIT CHECKER ──────────────────────────────────────────────────────

class UsageLimitChecker:
    """
    FastAPI dependency class that checks whether a user has remaining uses
    for a specific tool, based on their plan.

    Usage:
        @router.post("/tools/vocal-remover")
        async def vocal_remover(
            user: User = Depends(get_current_user),
            _limit: None = Depends(UsageLimitChecker("vocal_remover")),
        ):
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    async def __call__(
        self,
        request: Request,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> None:
        plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
        max_uses = plan["tool_uses_24h"]

        # Unlimited for max plan
        if max_uses == -1:
            await self._record_usage(user, db, request)
            return

        # Get or create usage record
        stmt = select(UsageLimit).where(
            UsageLimit.user_id == user.id,
            UsageLimit.tool_name == self.tool_name,
        )
        result = await db.execute(stmt)
        usage = result.scalar_one_or_none()

        now = datetime.utcnow()

        if usage:
            # Check if 24h window has elapsed → reset
            window_elapsed = now - usage.window_start
            if window_elapsed >= timedelta(hours=24):
                usage.usage_count_24h = 0
                usage.window_start = now

            if usage.usage_count_24h >= max_uses:
                reset_at = usage.window_start + timedelta(hours=24)
                retry_after = int((reset_at - now).total_seconds())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "message": f"Usage limit reached for {self.tool_name}. "
                                   f"You have used {usage.usage_count_24h}/{max_uses} "
                                   f"allowed uses in the last 24 hours.",
                        "tool": self.tool_name,
                        "used": usage.usage_count_24h,
                        "max": max_uses,
                        "retry_after": retry_after,
                        "reset_at": reset_at.isoformat(),
                        "upgrade_url": "/pricing",
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            usage.usage_count_24h += 1
            usage.last_used_at = now
        else:
            # Create new usage record
            device = await register_device(request, user, db)
            usage = UsageLimit(
                user_id=user.id,
                device_id=device.id,
                tool_name=self.tool_name,
                usage_count_24h=1,
                last_used_at=now,
                window_start=now,
            )
            db.add(usage)

        await db.flush()

    async def _record_usage(
        self, user: User, db: AsyncSession, request: Request
    ) -> None:
        """Record usage for unlimited plans (for analytics)."""
        stmt = select(UsageLimit).where(
            UsageLimit.user_id == user.id,
            UsageLimit.tool_name == self.tool_name,
        )
        result = await db.execute(stmt)
        usage = result.scalar_one_or_none()
        now = datetime.utcnow()

        if usage:
            if (now - usage.window_start) >= timedelta(hours=24):
                usage.usage_count_24h = 1
                usage.window_start = now
            else:
                usage.usage_count_24h += 1
            usage.last_used_at = now
        else:
            device = await register_device(request, user, db)
            usage = UsageLimit(
                user_id=user.id,
                device_id=device.id,
                tool_name=self.tool_name,
                usage_count_24h=1,
                last_used_at=now,
                window_start=now,
            )
            db.add(usage)
        await db.flush()


# ── YOUTUBE UPLOAD LIMIT CHECKER ──────────────────────────────────────────────

async def check_youtube_upload_limit(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Check monthly YouTube upload limit based on user's plan.
    Free = 5/month, Pro/Max = unlimited.
    """
    plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
    max_uploads = plan["youtube_uploads_monthly"]

    if max_uploads == -1:
        return  # unlimited

    today = datetime.utcnow().date()
    period_start = today.replace(day=1)

    stmt = select(YouTubeUploadMonthly).where(
        YouTubeUploadMonthly.user_id == user.id,
        YouTubeUploadMonthly.period_start == period_start,
    )
    result = await db.execute(stmt)
    monthly = result.scalar_one_or_none()

    if monthly and monthly.count >= max_uploads:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": f"Monthly YouTube upload limit reached ({monthly.count}/{max_uploads}). "
                           f"Upgrade to Pro for unlimited uploads.",
                "used": monthly.count,
                "max": max_uploads,
                "upgrade_url": "/pricing",
            },
        )


async def increment_youtube_upload_count(
    user: User,
    db: AsyncSession,
    device_id: Optional[str] = None,
) -> None:
    """Increment the monthly YouTube upload counter after successful upload."""
    today = datetime.utcnow().date()
    period_start = today.replace(day=1)

    stmt = select(YouTubeUploadMonthly).where(
        YouTubeUploadMonthly.user_id == user.id,
        YouTubeUploadMonthly.period_start == period_start,
    )
    result = await db.execute(stmt)
    monthly = result.scalar_one_or_none()

    if monthly:
        monthly.count += 1
    else:
        monthly = YouTubeUploadMonthly(
            user_id=user.id,
            count=1,
            period_start=period_start,
        )
        db.add(monthly)

    await db.flush()


# ── FILE SIZE CHECKER ─────────────────────────────────────────────────────────

def check_file_size(user: User, file_size_bytes: int) -> None:
    """
    Validate uploaded file size against the user's plan limit.
    """
    plan = PLAN_LIMITS.get(user.plan_type, PLAN_LIMITS["free"])
    max_mb = plan["max_file_size_mb"]
    max_bytes = max_mb * 1024 * 1024

    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "message": f"File too large. Max file size for {user.plan_type} plan is {max_mb}MB.",
                "max_mb": max_mb,
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "upgrade_url": "/pricing",
            },
        )


# ── MULTI-ACCOUNT ABUSE DETECTOR ─────────────────────────────────────────────

async def detect_abuse(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Middleware-style dependency that checks for multi-account abuse.
    Blocks if same IP is linked to >3 different accounts.
    """
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = _hash_value(client_ip)

    stmt = (
        select(func.count(func.distinct(Device.user_id)))
        .where(Device.ip_hash == ip_hash)
    )
    result = await db.execute(stmt)
    account_count = result.scalar() or 0

    if account_count > 3:
        logger.warning(
            f"Abuse detected: IP hash {ip_hash[:16]}... linked to {account_count} accounts"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Suspicious activity detected. Too many accounts from this network.",
        )
