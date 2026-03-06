"""
TuneVid.com — Payment Engine (Razorpay Integration)
Handles order creation, webhook verification, and subscription management.
"""
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import razorpay
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, Subscription, BillingEvent, Referral
from schemas import (
    CreateOrderRequest,
    CreateOrderResponse,
    PaymentVerifyRequest,
    SubscriptionOut,
    MessageResponse,
    PLAN_PRICING,
)

logger = logging.getLogger("tunevid.payments")

router = APIRouter(prefix="/api/payments", tags=["payments"])

# Razorpay client
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


def _detect_currency(request: Request) -> str:
    """Detect if user is from India based on headers → use INR, else USD."""
    # Check Cloudflare / proxy headers
    country = request.headers.get("CF-IPCountry", "")
    if not country:
        country = request.headers.get("X-Country-Code", "")

    # Accept-Language heuristic
    if not country:
        lang = request.headers.get("Accept-Language", "")
        if "hi" in lang or "en-IN" in lang:
            country = "IN"

    return "INR" if country == "IN" else "INR"  # Default to INR for now


# ── CREATE ORDER ──────────────────────────────────────────────────────────────

@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(
    body: CreateOrderRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Razorpay order for plan upgrade.
    Returns the order ID for frontend checkout.
    """
    plan = body.plan_type
    currency = _detect_currency(request)
    amount = PLAN_PRICING[plan][currency]

    try:
        order = razorpay_client.order.create({
            "amount": amount,
            "currency": currency,
            "receipt": f"tunevid_{user.id}_{plan}",
            "notes": {
                "user_id": str(user.id),
                "plan_type": plan,
                "email": user.email,
            },
        })
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway error. Please try again.",
        )

    # Store subscription record
    sub = Subscription(
        user_id=user.id,
        razorpay_order_id=order["id"],
        plan_type=plan,
        amount_paise=amount,
        currency=currency,
        status="created",
    )
    db.add(sub)
    await db.flush()

    return CreateOrderResponse(
        order_id=order["id"],
        amount=amount,
        currency=currency,
        key_id=RAZORPAY_KEY_ID,
        plan_type=plan,
        user_email=user.email,
        user_name=user.name,
    )


# ── VERIFY PAYMENT (Client-side verification) ────────────────────────────────

@router.post("/verify-payment", response_model=MessageResponse)
async def verify_payment(
    body: PaymentVerifyRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Verify a payment after Razorpay checkout completes on the client.
    """
    try:
        razorpay_client.utility.verify_payment_signature({
            "razorpay_order_id": body.razorpay_order_id,
            "razorpay_payment_id": body.razorpay_payment_id,
            "razorpay_signature": body.razorpay_signature,
        })
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed.",
        )

    # Update subscription
    stmt = select(Subscription).where(
        Subscription.razorpay_order_id == body.razorpay_order_id,
        Subscription.user_id == user.id,
    )
    result = await db.execute(stmt)
    sub = result.scalar_one_or_none()

    if not sub:
        raise HTTPException(status_code=404, detail="Order not found")

    sub.razorpay_payment_id = body.razorpay_payment_id
    sub.status = "paid"

    # Upgrade user plan
    user.plan_type = sub.plan_type
    user.subscription_end_date = datetime.utcnow() + timedelta(days=30)

    await db.flush()

    # Check if this user was referred → mark referral as qualified
    await _process_referral_reward(user, db)

    return MessageResponse(
        message=f"Payment verified! Upgraded to {sub.plan_type.upper()} plan.",
    )


# ── RAZORPAY WEBHOOK ─────────────────────────────────────────────────────────

@router.post("/webhook/razorpay")
async def razorpay_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Razorpay webhook handler — verifies signature, processes payment events.
    Handles: order.paid, payment.captured, payment.failed
    """
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    # Verify webhook signature
    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        logger.warning("Razorpay webhook signature verification failed")
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body)
    event_type = payload.get("event", "")
    event_id = payload.get("id", "")

    # Idempotency check
    existing = await db.execute(
        select(BillingEvent).where(BillingEvent.razorpay_event_id == event_id)
    )
    if existing.scalar_one_or_none():
        return {"status": "already_processed"}

    # Store the event
    billing_event = BillingEvent(
        razorpay_event_id=event_id,
        event_type=event_type,
        payload=payload,
    )
    db.add(billing_event)

    # Process based on event type
    if event_type == "order.paid":
        await _handle_order_paid(payload, db)
        billing_event.processed = True
    elif event_type == "payment.failed":
        await _handle_payment_failed(payload, db)
        billing_event.processed = True

    await db.flush()
    return {"status": "ok"}


async def _handle_order_paid(payload: dict, db: AsyncSession):
    """Process order.paid webhook event."""
    try:
        order_entity = payload["payload"]["order"]["entity"]
        order_id = order_entity["id"]
        notes = order_entity.get("notes", {})
        user_id = notes.get("user_id")
        plan_type = notes.get("plan_type", "pro")

        if not user_id:
            logger.error(f"order.paid missing user_id in notes: {order_id}")
            return

        # Find subscription
        stmt = select(Subscription).where(
            Subscription.razorpay_order_id == order_id
        )
        result = await db.execute(stmt)
        sub = result.scalar_one_or_none()

        if sub:
            payment_entity = payload["payload"].get("payment", {}).get("entity", {})
            sub.razorpay_payment_id = payment_entity.get("id") or sub.razorpay_payment_id
            sub.status = "paid"

        # Update user plan
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            user.plan_type = plan_type
            user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
            logger.info(f"User {user.email} upgraded to {plan_type} via webhook")

            # Process referral
            await _process_referral_reward(user, db)

    except Exception as e:
        logger.error(f"Error processing order.paid: {e}")


async def _handle_payment_failed(payload: dict, db: AsyncSession):
    """Process payment.failed webhook event."""
    try:
        payment_entity = payload["payload"]["payment"]["entity"]
        order_id = payment_entity.get("order_id")

        if order_id:
            stmt = select(Subscription).where(
                Subscription.razorpay_order_id == order_id
            )
            result = await db.execute(stmt)
            sub = result.scalar_one_or_none()
            if sub:
                sub.status = "failed"
                logger.info(f"Payment failed for order {order_id}")

    except Exception as e:
        logger.error(f"Error processing payment.failed: {e}")


async def _process_referral_reward(user: User, db: AsyncSession):
    """
    If this user was referred, mark the referral as qualified
    and reward the referrer with 1 month Pro.
    """
    stmt = select(Referral).where(
        Referral.referee_id == user.id,
        Referral.status == "pending",
    )
    result = await db.execute(stmt)
    referral = result.scalar_one_or_none()

    if referral:
        referral.status = "qualified"

        # Find referrer
        stmt = select(User).where(User.id == referral.referrer_id)
        result = await db.execute(stmt)
        referrer = result.scalar_one_or_none()

        if referrer and not referral.reward_applied:
            # Give referrer 1 month of Pro
            if referrer.plan_type == "free":
                referrer.plan_type = "pro"
                referrer.subscription_end_date = datetime.utcnow() + timedelta(days=30)
            elif referrer.subscription_end_date:
                referrer.subscription_end_date += timedelta(days=30)
            else:
                referrer.subscription_end_date = datetime.utcnow() + timedelta(days=30)

            referral.reward_applied = True
            referral.status = "rewarded"
            logger.info(f"Referral reward: {referrer.email} got 1mo Pro for referring {user.email}")

    await db.flush()


# ── GET SUBSCRIPTIONS ─────────────────────────────────────────────────────────

@router.get("/subscriptions", response_model=list[SubscriptionOut])
async def get_subscriptions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all subscriptions for the current user."""
    stmt = (
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.created_at.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    return result.scalars().all()
