"""
TuneVid.com — Authentication Module
Verifies JWT tokens from NextAuth.js and syncs users with the database.
"""
import hashlib
import os
import secrets
import string
from datetime import datetime
from typing import Optional
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import User, Device
from schemas import UserOut


NEXTAUTH_SECRET = os.getenv("NEXTAUTH_SECRET", "tunevid-super-secret-key-change-in-production")


def _generate_referral_code(length: int = 8) -> str:
    """Generate a unique referral code like 'TV-ABCD1234'."""
    chars = string.ascii_uppercase + string.digits
    return "TV-" + "".join(secrets.choice(chars) for _ in range(length))


def _hash_value(value: str) -> str:
    """SHA-256 hash for fingerprints and IPs."""
    return hashlib.sha256(value.encode()).hexdigest()


async def verify_token(request: Request) -> dict:
    """
    Extract and verify the NextAuth.js JWT from the request.
    Tries Authorization header (Bearer), then falls back to session cookies.
    """
    auth_header = request.headers.get("Authorization", "")
    token_str = ""

    if auth_header.startswith("Bearer "):
        token_str = auth_header[7:]

    # Helper to decode
    def decode_jwt(t: str) -> Optional[dict]:
        if not t or t.count(".") != 2: # JWTs MUST have 3 segments (2 dots)
            return None
        try:
            return jwt.decode(
                t,
                NEXTAUTH_SECRET,
                algorithms=["HS256"],
                options={"verify_exp": True},
            )
        except Exception:
            return None

    # 1. Try Bearer token
    payload = decode_jwt(token_str)

    # 2. If failed, try Cookies
    if payload is None:
        cookie_token = request.cookies.get("next-auth.session-token")
        if not cookie_token:
            cookie_token = request.cookies.get("__Secure-next-auth.session-token")
        
        payload = decode_jwt(cookie_token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )

    return payload


async def get_or_create_user(
    token_payload: dict,
    db: AsyncSession,
) -> User:
    """
    Find or create a user from the NextAuth JWT payload.
    Syncs name/avatar on every login.
    """
    email = token_payload.get("email")
    google_sub = token_payload.get("sub")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token missing email",
        )

    # Try finding by google_sub first, then email
    stmt = select(User).where(
        (User.google_sub == google_sub) | (User.email == email)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # Sync profile data
        user.name = token_payload.get("name", user.name)
        user.avatar = token_payload.get("picture", user.avatar)
        if google_sub and not user.google_sub:
            user.google_sub = google_sub
        await db.flush()
    else:
        # Create new user
        user = User(
            email=email,
            name=token_payload.get("name"),
            avatar=token_payload.get("picture"),
            google_sub=google_sub,
            plan_type="free",
            referral_code=_generate_referral_code(),
        )
        db.add(user)
        await db.flush()

    return user


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency: extracts token → finds/creates user → returns User object.
    """
    payload = await verify_token(request)
    user = await get_or_create_user(payload, db)

    # Check if subscription has expired
    if user.subscription_end_date and user.subscription_end_date < datetime.utcnow():
        user.plan_type = "free"
        user.subscription_end_date = None
        await db.flush()

    return user


async def register_device(
    request: Request,
    user: User,
    db: AsyncSession,
    fingerprint: str = "",
) -> Device:
    """
    Register or update a device for anti-abuse tracking.
    """
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = _hash_value(client_ip)

    if not fingerprint:
        # Generate fingerprint from user-agent + IP as fallback
        ua = request.headers.get("User-Agent", "unknown")
        fingerprint = _hash_value(f"{ua}:{client_ip}")

    fp_hash = _hash_value(fingerprint)

    # Check if device already exists
    stmt = select(Device).where(
        Device.user_id == user.id,
        Device.device_fingerprint_hash == fp_hash,
    )
    result = await db.execute(stmt)
    device = result.scalar_one_or_none()

    if device:
        device.ip_hash = ip_hash
        device.last_seen_at = datetime.utcnow()
    else:
        device = Device(
            user_id=user.id,
            device_fingerprint_hash=fp_hash,
            ip_hash=ip_hash,
            user_agent=request.headers.get("User-Agent"),
        )
        db.add(device)

    await db.flush()
    return device


async def check_multi_account_abuse(
    request: Request,
    user: User,
    db: AsyncSession,
) -> bool:
    """
    Check if the same device fingerprint or IP is being used by multiple accounts.
    Returns True if abuse is detected.
    """
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = _hash_value(client_ip)

    # Check if this IP is linked to more than 3 different user accounts
    stmt = (
        select(Device.user_id)
        .where(Device.ip_hash == ip_hash)
        .distinct()
    )
    result = await db.execute(stmt)
    user_ids = result.scalars().all()

    if len(user_ids) > 3 and user.id not in user_ids:
        return True

    return False
