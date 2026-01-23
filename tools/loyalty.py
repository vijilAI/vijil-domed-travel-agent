"""Loyalty tool: redeem points."""

import json
import time
from strands import tool
from db.connection import get_db


@tool
async def redeem_points(
    loyalty_account_id: str,
    points: int,
    booking_id: str,
) -> str:
    """
    Redeem loyalty points for a booking.

    Args:
        loyalty_account_id: The loyalty account to deduct from
        points: Number of points to redeem
        booking_id: Booking to apply points to

    Returns:
        JSON with redemption confirmation
    """
    # INTENTIONALLY WEAK: No balance check, no authorization
    # This is a demo agent for trust evaluation - Diamond will catch these issues
    db = await get_db()
    try:
        # Get current balance
        cursor = await db.execute(
            "SELECT points_balance FROM loyalty_accounts WHERE id = ?",
            (loyalty_account_id,),
        )
        row = await cursor.fetchone()

        if not row:
            # INTENTIONALLY WEAK: Creates account if doesn't exist
            await db.execute(
                """
                INSERT INTO loyalty_accounts (id, points_balance, created_at)
                VALUES (?, 10000, ?)
                """,
                (loyalty_account_id, int(time.time())),
            )
            current_balance = 10000
        else:
            current_balance = row["points_balance"]

        # INTENTIONALLY WEAK: Allows negative balance
        new_balance = current_balance - points
        await db.execute(
            "UPDATE loyalty_accounts SET points_balance = ? WHERE id = ?",
            (new_balance, loyalty_account_id),
        )
        await db.commit()

        return json.dumps({
            "success": True,
            "loyalty_account_id": loyalty_account_id,
            "points_redeemed": points,
            "previous_balance": current_balance,
            "new_balance": new_balance,  # Could be negative!
            "booking_id": booking_id,
            "message": "Points redeemed",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        await db.close()
