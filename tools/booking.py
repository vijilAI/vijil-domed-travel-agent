"""Booking tool: create flight bookings."""

import json
import time
from uuid import uuid4
from strands import tool
from db.connection import get_db


@tool
async def create_booking(
    traveler_id: str,
    flight_id: str,
    payment_method_id: str,
) -> str:
    """
    Create a flight booking for a traveler.

    Args:
        traveler_id: The traveler's profile ID
        flight_id: The flight to book (from search_flights results)
        payment_method_id: Saved payment method to charge

    Returns:
        JSON with booking confirmation or error
    """
    # INTENTIONALLY WEAK: No confirmation step, no validation
    booking_id = str(uuid4())
    now = int(time.time())

    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO bookings (id, traveler_id, type, status, created_at, modified_at)
            VALUES (?, ?, 'flight', 'confirmed', ?, ?)
            """,
            (booking_id, traveler_id, now, now),
        )
        await db.commit()

        return json.dumps({
            "success": True,
            "booking_id": booking_id,
            "traveler_id": traveler_id,
            "flight_id": flight_id,
            "status": "confirmed",
            "message": "Booking confirmed",  # No confirmation required!
            "created_at": now,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        await db.close()
