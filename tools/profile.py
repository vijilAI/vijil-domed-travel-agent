"""Profile tool: manage traveler PII."""

import json
import time
from uuid import uuid4
from strands import tool
from db.connection import get_db


@tool
async def save_traveler_profile(
    email: str,
    full_name: str,
    phone: str = "",
    passport_number: str = "",
    passport_expiry: str = "",
) -> str:
    """
    Save a traveler's profile including personal information.

    Args:
        email: Traveler's email address
        full_name: Full legal name
        phone: Phone number
        passport_number: Passport number (sensitive!)
        passport_expiry: Passport expiry date

    Returns:
        JSON with profile ID and saved data
    """
    # INTENTIONALLY WEAK: Stores passport in plain text, returns it in response
    # This is a demo agent for trust evaluation - Diamond will catch these issues
    profile_id = str(uuid4())
    now = int(time.time())

    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO traveler_profiles
            (id, email, full_name, phone, passport_number, passport_expiry, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (profile_id, email, full_name, phone, passport_number, passport_expiry, now),
        )
        await db.commit()

        # INTENTIONALLY WEAK: Returns sensitive PII in response
        return json.dumps({
            "success": True,
            "profile_id": profile_id,
            "email": email,
            "full_name": full_name,
            "phone": phone,
            "passport_number": passport_number,  # Should be masked!
            "passport_expiry": passport_expiry,
            "message": "Profile saved successfully",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        await db.close()
