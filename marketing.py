"""
TuneVid.com — Marketing & Referral Engine
Handles referral codes, auto-promo for free users, and viral mechanics.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, Referral
from schemas import (
    ApplyReferralRequest,
    ReferralOut,
    MessageResponse,
)

logger = logging.getLogger("tunevid.marketing")

router = APIRouter(prefix="/api/marketing", tags=["marketing"])

# ── AUTO-PROMO LOGIC ─────────────────────────────────────────────────────────

AUTO_PROMO_TEXT = (
    "\n\n✨ Created with TuneVid.com — FREE AI Audio Studio & YouTube Uploader 🎵\n"
    "🔗 https://tunevid.com | Remove Vocals, Boost Bass, Create 8D Audio & More!\n"
    "#TuneVid #AudioTools #YouTubeCreator"
)


def inject_auto_promo(description: str, plan_type: str) -> str:
    """
    For free users, append auto-promo to YouTube video description.
    Pro and Max users get clean descriptions.
    """
    if plan_type == "free":
        # Don't double-add if already present
        if "tunevid.com" not in description.lower():
            return description + AUTO_PROMO_TEXT
    return description


# ── REFERRAL ENDPOINTS ────────────────────────────────────────────────────────

@router.get("/referral-code")
async def get_my_referral_code(
    user: User = Depends(get_current_user),
):
    """Get the current user's referral code."""
    return {
        "referral_code": user.referral_code,
        "referral_link": f"https://tunevid.com/?ref={user.referral_code}",
    }


@router.post("/apply-referral", response_model=MessageResponse)
async def apply_referral(
    body: ApplyReferralRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Apply a referral code. The referee (current user) links to the referrer.
    Referrer gets 1 month Pro when referee either:
    - Uploads their first video, OR
    - Buys a plan.
    """
    code = body.referral_code.strip().upper()

    # Find referrer by code
    stmt = select(User).where(User.referral_code == code)
    result = await db.execute(stmt)
    referrer = result.scalar_one_or_none()

    if not referrer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid referral code.",
        )

    if referrer.id == user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can't use your own referral code.",
        )

    # Check if already referred
    existing = await db.execute(
        select(Referral).where(Referral.referee_id == user.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already used a referral code.",
        )

    # Create referral record
    referral = Referral(
        referrer_id=referrer.id,
        referee_id=user.id,
        status="pending",
    )
    db.add(referral)
    await db.flush()

    return MessageResponse(
        message=f"Referral applied! {referrer.name or 'Your friend'} will be rewarded when you upload or subscribe.",
    )


@router.get("/my-referrals", response_model=list[ReferralOut])
async def get_my_referrals(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get list of people the current user has referred."""
    stmt = (
        select(Referral)
        .where(Referral.referrer_id == user.id)
        .order_by(Referral.created_at.desc())
    )
    result = await db.execute(stmt)
    referrals = result.scalars().all()

    out = []
    for ref in referrals:
        # Fetch referee info
        referee_stmt = select(User).where(User.id == ref.referee_id)
        referee_result = await db.execute(referee_stmt)
        referee = referee_result.scalar_one_or_none()

        out.append(ReferralOut(
            id=ref.id,
            referee_email=referee.email if referee else None,
            referee_name=referee.name if referee else None,
            status=ref.status,
            reward_applied=ref.reward_applied,
            created_at=ref.created_at,
        ))

    return out


@router.get("/referral-stats")
async def get_referral_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get referral statistics for the dashboard."""
    # Total referrals
    total_stmt = select(func.count()).where(Referral.referrer_id == user.id)
    total_result = await db.execute(total_stmt)
    total = total_result.scalar() or 0

    # Rewarded count
    rewarded_stmt = select(func.count()).where(
        Referral.referrer_id == user.id,
        Referral.status == "rewarded",
    )
    rewarded_result = await db.execute(rewarded_stmt)
    rewarded = rewarded_result.scalar() or 0

    # Pending count
    pending_stmt = select(func.count()).where(
        Referral.referrer_id == user.id,
        Referral.status == "pending",
    )
    pending_result = await db.execute(pending_stmt)
    pending = pending_result.scalar() or 0

    return {
        "referral_code": user.referral_code,
        "referral_link": f"https://tunevid.com/?ref={user.referral_code}",
        "total_referrals": total,
        "rewarded_referrals": rewarded,
        "pending_referrals": pending,
        "total_pro_months_earned": rewarded,
    }


# ── REFERRAL QUALIFICATION ON FIRST UPLOAD ────────────────────────────────────

async def qualify_referral_on_upload(user: User, db: AsyncSession):
    """
    Called after a user's first YouTube upload.
    If they were referred, mark the referral as qualified and reward referrer.
    """
    stmt = select(Referral).where(
        Referral.referee_id == user.id,
        Referral.status == "pending",
    )
    result = await db.execute(stmt)
    referral = result.scalar_one_or_none()

    if not referral:
        return

    referral.status = "qualified"

    # Reward the referrer
    referrer_stmt = select(User).where(User.id == referral.referrer_id)
    referrer_result = await db.execute(referrer_stmt)
    referrer = referrer_result.scalar_one_or_none()

    if referrer and not referral.reward_applied:
        if referrer.plan_type == "free":
            referrer.plan_type = "pro"
            referrer.subscription_end_date = datetime.utcnow() + timedelta(days=30)
        elif referrer.subscription_end_date:
            referrer.subscription_end_date += timedelta(days=30)
        else:
            referrer.subscription_end_date = datetime.utcnow() + timedelta(days=30)

        referral.reward_applied = True
        referral.status = "rewarded"
        logger.info(
            f"Referral reward (upload): {referrer.email} got 1mo Pro "
            f"for referring {user.email}"
        )

    await db.flush()
