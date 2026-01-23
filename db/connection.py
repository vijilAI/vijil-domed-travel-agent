"""Database connection management using aiosqlite."""

import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "travel_agent.db"


async def get_db() -> aiosqlite.Connection:
    """Get a database connection."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db() -> None:
    """Initialize database tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Traveler profiles
        await db.execute("""
            CREATE TABLE IF NOT EXISTS traveler_profiles (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                full_name TEXT,
                phone TEXT,
                passport_number TEXT,
                passport_expiry TEXT,
                loyalty_program_id TEXT,
                created_at INTEGER
            )
        """)

        # Bookings
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id TEXT PRIMARY KEY,
                traveler_id TEXT REFERENCES traveler_profiles(id),
                type TEXT,
                status TEXT,
                origin TEXT,
                destination TEXT,
                departure_at INTEGER,
                return_at INTEGER,
                flight_number TEXT,
                hotel_name TEXT,
                total_amount REAL,
                currency TEXT DEFAULT 'USD',
                created_at INTEGER,
                modified_at INTEGER
            )
        """)

        # Payments
        await db.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id TEXT PRIMARY KEY,
                booking_id TEXT REFERENCES bookings(id),
                amount REAL,
                currency TEXT DEFAULT 'USD',
                card_last_four TEXT,
                status TEXT,
                created_at INTEGER
            )
        """)

        # Loyalty accounts
        await db.execute("""
            CREATE TABLE IF NOT EXISTS loyalty_accounts (
                id TEXT PRIMARY KEY,
                traveler_id TEXT REFERENCES traveler_profiles(id),
                program_name TEXT,
                points_balance INTEGER,
                created_at INTEGER
            )
        """)

        # Expenses
        await db.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id TEXT PRIMARY KEY,
                traveler_id TEXT REFERENCES traveler_profiles(id),
                booking_id TEXT REFERENCES bookings(id),
                amount REAL,
                category TEXT,
                receipt_url TEXT,
                status TEXT,
                created_at INTEGER
            )
        """)

        # Policy rules
        await db.execute("""
            CREATE TABLE IF NOT EXISTS policy_rules (
                id TEXT PRIMARY KEY,
                rule_type TEXT,
                rule_value TEXT,
                active INTEGER DEFAULT 1
            )
        """)

        await db.commit()


if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
    print(f"Database initialized at {DB_PATH}")
