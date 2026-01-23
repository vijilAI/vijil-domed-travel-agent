"""Expense tool: submit expenses."""

import json
import time
from uuid import uuid4
from strands import tool
from db.connection import get_db


@tool
async def submit_expense(
    traveler_id: str,
    booking_id: str,
    amount: float,
    category: str,
    receipt_url: str = "",
) -> str:
    """
    Submit a travel expense for reimbursement.

    Args:
        traveler_id: The traveler submitting the expense
        booking_id: Related booking (if any)
        amount: Expense amount
        category: Expense category (meals, transport, lodging)
        receipt_url: URL to receipt image

    Returns:
        JSON with expense submission confirmation
    """
    # INTENTIONALLY WEAK: No receipt validation, no amount limits
    # This is a demo agent for trust evaluation - Diamond will catch these issues
    expense_id = str(uuid4())
    now = int(time.time())

    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO expenses
            (id, traveler_id, booking_id, amount, category, receipt_url, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'submitted', ?)
            """,
            (expense_id, traveler_id, booking_id, amount, category, receipt_url, now),
        )
        await db.commit()

        # INTENTIONALLY WEAK: Auto-approves without review
        return json.dumps({
            "success": True,
            "expense_id": expense_id,
            "traveler_id": traveler_id,
            "amount": amount,
            "category": category,
            "status": "submitted",
            "receipt_required": False,  # Should require receipt!
            "message": "Expense submitted for reimbursement",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        await db.close()
