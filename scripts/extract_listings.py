"""Turn raw Willhaben search-API responses into the slim listing sample used by the pipeline.

Input : data/raw/willhaben/*.json   (git-ignored; copied manually from the browser's network tab)
Output: data/listings.json          (committed; only coordinates, price, size, rooms, type)

Free text, images and advertiser information are dropped on purpose so the
committed sample contains no personal data.  Listings that are obviously not
long-term flats (garages, parking spots, offices, per-day/per-week rentals)
are removed here because that can only be decided from the listing title.

Usage:  python scripts/extract_listings.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "willhaben"
OUTPUT = ROOT / "data" / "listings.json"

# Titles matching these patterns are not residential long-term rentals.
EXCLUDE_TITLE = re.compile(
    r"garage|stellplatz|parkplatz|kfz|tiefgarage|lager|büro|geschäftslokal|nagelstudio"
    r"|pro tag|pro woche|kurzzeit|notfallwohnung|arbeiterquartier|monteur",
    re.IGNORECASE,
)


def _attributes(advert: dict) -> dict[str, str]:
    return {a["name"]: "|".join(a["values"]) for a in advert["attributes"]["attribute"]}


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def extract() -> list[dict]:
    listings, skipped = [], 0
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for advert in data["advertSummaryList"]["advertSummary"]:
            attrs = _attributes(advert)
            if "COORDINATES" not in attrs or EXCLUDE_TITLE.search(attrs.get("HEADING", "")):
                skipped += 1
                continue
            lat, lon = (float(x) for x in attrs["COORDINATES"].split(","))
            listings.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "price": _float(attrs.get("PRICE")),
                "size": _float(attrs.get("ESTATE_SIZE/LIVING_AREA") or attrs.get("ESTATE_SIZE")),
                "rooms": _float(attrs.get("NUMBER_OF_ROOMS")),
                # the raw dumps contain U+FFFD where "ß" used to be (e.g. "Dachgeschoßwohnung")
                "type": attrs.get("PROPERTY_TYPE", "").replace("\ufffd", "ß"),
                "postcode": attrs.get("POSTCODE", ""),
            })
    print(f"Kept {len(listings)} listings, skipped {skipped} (no coordinates or not a long-term flat)")
    return listings


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(extract(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved to {OUTPUT.relative_to(ROOT)}")
