"""
tools.py — Mock airline data tools.

All tools return fixed fake data so every test run produces identical results.
search_flights intentionally returns only a Business-class seat for CDG→JFK
to trigger the policy-checker retry loop in Scenario 3.
"""
from langchain_core.tools import tool

# ── Mock databases ────────────────────────────────────────────────────────────

BOOKINGS = {
    # Scenario 1 — cancelled, Economy, refundable → rebook succeeds
    "XK9L2P": {
        "passenger": "John Doe",
        "route": "LHR-DXB",
        "cabin_class": "Economy",
        "fare_type": "refundable",
        "disruption_details": "flight cancelled by airline",
        "origin": "LHR",
        "destination": "DXB",
    },
    # Scenario 2 — 6-hour delay → compensation $200
    "B77XYZ": {
        "passenger": "Jane Smith",
        "route": "JFK-LHR",
        "cabin_class": "Business",
        "fare_type": "non-refundable",
        "disruption_details": "flight delayed 6 hours",
        "origin": "JFK",
        "destination": "LHR",
    },
    # Scenario 3 — cancelled, Economy → only Business available → 3 policy fails → escalation
    "C88ABC": {
        "passenger": "Bob Jones",
        "route": "CDG-JFK",
        "cabin_class": "Economy",
        "fare_type": "refundable",
        "disruption_details": "flight cancelled by airline",
        "origin": "CDG",
        "destination": "JFK",
    },
    # Scenario 4 — complaint → straight to escalation
    "D99DEF": {
        "passenger": "Alice Brown",
        "route": "SYD-LAX",
        "cabin_class": "Economy",
        "fare_type": "non-refundable",
        "disruption_details": "flight cancelled by airline (third time)",
        "origin": "SYD",
        "destination": "LAX",
    },
}

FLIGHTS = [
    # Scenario 1: LHR→DXB — Economy cabin, 120-min connection → passes policy
    {
        "flight_no": "EK001",
        "departure": "08:00",
        "arrival": "18:00",
        "cabin": "Economy",
        "connection_time": 120,
        "departs_in_hours": 14,
        "origin": "LHR",
        "destination": "DXB",
    },
    # Scenario 3: CDG→JFK — only Business cabin available → ALWAYS fails policy (cabin upgrade)
    {
        "flight_no": "AF100",
        "departure": "10:00",
        "arrival": "13:00",
        "cabin": "Business",
        "connection_time": 90,
        "departs_in_hours": 20,
        "origin": "CDG",
        "destination": "JFK",
    },
]

FARE_RULES = {
    "refundable": {"refundable": True, "tax_amount": 50},
    "non-refundable": {"refundable": False, "tax_amount": 75},
}

# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def get_booking(booking_ref: str) -> dict:
    """
    Retrieve booking details for a passenger.
    Returns passenger name, route, cabin_class, fare_type, and disruption_details.
    """
    return BOOKINGS.get(booking_ref, {"error": f"Booking '{booking_ref}' not found."})


@tool
def search_flights(origin: str, destination: str, date: str) -> list:
    """
    Search for available flights between two airports on a given date.
    Returns a list of flights with flight_no, departure, arrival, cabin, connection_time,
    departs_in_hours, origin, and destination.
    """
    results = [
        f for f in FLIGHTS
        if f["origin"].upper() == origin.upper()
        and f["destination"].upper() == destination.upper()
    ]
    if not results:
        return [{"error": f"No flights found from {origin} to {destination} on {date}."}]
    return results


@tool
def get_fare_rules(fare_type: str) -> dict:
    """
    Retrieve fare rules for a given fare type.
    Returns whether the fare is refundable and the tax amount in USD.
    """
    return FARE_RULES.get(fare_type, {"error": f"Fare type '{fare_type}' not found."})
