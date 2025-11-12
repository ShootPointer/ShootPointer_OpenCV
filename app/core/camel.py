#app/core/camel.py
def to_camel(s: str) -> str:
    # "jersey_number" -> "jerseyNumber"
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])
