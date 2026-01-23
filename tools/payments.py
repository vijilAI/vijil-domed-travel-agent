"""Payment tool: process payments."""

import json
import time
from uuid import uuid4
from strands import tool
from db.connection import get_db


@tool
async def process_payment(
    booking_id: str,
    amount: float,
    card_last_four: str,
) -> str:
    """
    Process a payment for a booking.

    Args:
        booking_id: The booking to pay for
        amount: Payment amount
        card_last_four: Last 4 digits of card

    Returns:
        JSON with payment confirmation
    """
    # INTENTIONALLY WEAK: No amount validation, no fraud checks
    # This is a demo agent for trust evaluation - Diamond will catch these issues
    payment_id = str(uuid4())
    now = int(time.time())

    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO payments (id, booking_id, amount, card_last_four, status, created_at)
            VALUES (?, ?, ?, ?, 'completed', ?)
            """,
            (payment_id, booking_id, amount, card_last_four, now),
        )
        await db.commit()

        # INTENTIONALLY WEAK: Processes any amount without confirmation
        return json.dumps({
            "success": True,
            "payment_id": payment_id,
            "booking_id": booking_id,
            "amount": amount,
            "currency": "USD",
            "card_last_four": card_last_four,
            "status": "completed",
            "message": "Payment processed",  # No confirmation step!
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        await db.close()
