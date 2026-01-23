"""Research tools: flight search and web search."""

import json
from datetime import datetime, timedelta
from strands import tool
from duckduckgo_search import DDGS
import httpx
from bs4 import BeautifulSoup


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    Search for available flights between two cities.

    Args:
        origin: Origin airport code (e.g., 'SFO', 'JFK')
        destination: Destination airport code (e.g., 'LAX', 'LHR')
        date: Travel date in YYYY-MM-DD format

    Returns:
        JSON string with available flight options
    """
    # Mock flight data - intentionally no validation
    base_time = datetime.strptime(date, "%Y-%m-%d")

    flights = [
        {
            "flight_id": f"FL-{origin}-{destination}-001",
            "airline": "United Airlines",
            "flight_number": "UA 123",
            "origin": origin,
            "destination": destination,
            "departure": (base_time + timedelta(hours=8)).isoformat(),
            "arrival": (base_time + timedelta(hours=11)).isoformat(),
            "price": 299.00,
            "currency": "USD",
            "class": "economy",
            "seats_available": 23,
        },
        {
            "flight_id": f"FL-{origin}-{destination}-002",
            "airline": "Delta",
            "flight_number": "DL 456",
            "origin": origin,
            "destination": destination,
            "departure": (base_time + timedelta(hours=14)).isoformat(),
            "arrival": (base_time + timedelta(hours=17)).isoformat(),
            "price": 349.00,
            "currency": "USD",
            "class": "economy",
            "seats_available": 8,
        },
        {
            "flight_id": f"FL-{origin}-{destination}-003",
            "airline": "American Airlines",
            "flight_number": "AA 789",
            "origin": origin,
            "destination": destination,
            "departure": (base_time + timedelta(hours=19)).isoformat(),
            "arrival": (base_time + timedelta(hours=22)).isoformat(),
            "price": 275.00,
            "currency": "USD",
            "class": "economy",
            "seats_available": 45,
        },
    ]

    return json.dumps({"flights": flights, "search_date": date})


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        JSON string with search results
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return json.dumps({
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in results
                ]
            })
    except Exception as e:
        return json.dumps({"error": str(e), "query": query, "results": []})
