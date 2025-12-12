# Parses basic information from a project description string

import re
from dataclasses import dataclass


@dataclass
class ParsedDescription:
    text: str
    street_address: str | None
    borough: str | None
    disturbed_area_sf: float | None
    new_impervious_sf: float | None


# Regex patterns used to extract values from text

# Square footage numbers like "12,000 SF" or "12000 sq ft"
NUMBER_SF_PATTERN = re.compile(
    r"([\d,]+)\s*(?:sf|square\s*feet|sq\s*ft)",
    flags=re.IGNORECASE
)

# Address pattern like "123 Main Street", "460 New Dorp Lane", "116 3rd Avenue"
ADDRESS_PATTERN = re.compile(
    r"\b("                          # start capture
    r"\d+\s+"                           # street number
    r"[A-Za-z0-9.\-'\s]+?"              # street name tokens
    r"\s+"                              # space before suffix
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Road|Rd|Drive|Dr)"  # suffix
    r")\b",                             # stop capture at end of suffix
    flags=re.IGNORECASE
)

# Borough pattern like "in the borough of Brooklyn"
BOROUGH_PATTERN = re.compile(
    r"\b(?:in\s+(?:the\s+borough\s+of\s+)?)?"
    r"(Bronx|Brooklyn|Queens|Manhattan|Staten\s+Island|SI|S\.I\.)\b",
    flags=re.IGNORECASE
)


# Main parsing function
def parse_description(text: str) -> ParsedDescription:
    """
    Extract:
      - Address (string)
      - Disturbed area (numeric)
      - New impervious area (numeric)
    """

    # Extract address and borough if present
    address_match = ADDRESS_PATTERN.search(text)
    borough_match = BOROUGH_PATTERN.search(text)

    if address_match:
        street_address = address_match.group(1)
    else:
        street_address = None

    borough = borough_match.group(1) if borough_match else None

    # Try to find disturbed area
    disturbed_area = None

    # Look for phrases implying disturbance:
    DISTURB_PHRASES = [
        r"disturb(?:s|ed|ance|ing)?\s*(?:approximately|around|roughly)?\s*([\d,]+\s*(?:sf|square\s*feet|sq\s*ft))",
        r"soil\s+disturbance\s*(?:of)?\s*([\d,]+\s*(?:sf|square\s*feet))",
    ]

    for pat in DISTURB_PHRASES:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            disturbed_area = _extract_sf_number(m.group(1))
            break

    # If there is only one square footage mentioned, assume it refers to disturbance
    if disturbed_area is None:
        all_sf_numbers = NUMBER_SF_PATTERN.findall(text)
        if len(all_sf_numbers) == 1:
            disturbed_area = _extract_sf_number(all_sf_numbers[0])

    # Check for phrases that imply the entire site is disturbed
    FULL_SITE_PHRASES = [
        r"disturb(?:ing|s|ed)?\s+(?:the\s+)?(?:entire|full)\s+(?:site|lot|parcel)",
        r"(?:entire|full)\s+(?:site|lot|parcel)\s+will\s+be\s+disturbed",
        r"full[-\s]?site",
        r"full[-\s]?lot",
        r"full[-\s]?parcel",
        r"full[-\s]depth\s+reconstruction",
        r"the\s+entire\s+.*\s+will\s+be\s+disturbed",
    ]

    # Fall back to full lot area
    if disturbed_area is None:
        for pat in FULL_SITE_PHRASES:
            if re.search(pat, text, re.IGNORECASE):
                disturbed_area = "FULL_SITE"
                break

    # Try to determine new impervious area
    new_imp = None

    IMPERVIOUS_PHRASES = [
        r"new\s+impervious\s+area\s*(?:of)?\s*([\d,]+\s*(?:sf|square\s*feet))",
        r"adding\s*([\d,]+\s*(?:sf|square\s*feet))\s+of\s+new\s+impervious",
        r"(?:propos(?:es|ing)\s*)?([\d,]+\s*(?:sf|square\s*feet))\s*(?:of)?\s*new\s+impervious",
    ]

    for pat in IMPERVIOUS_PHRASES:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            new_imp = _extract_sf_number(m.group(1))
            break

    # If a new building is mentioned, assume new impervious area
    NEW_BUILDING_PATTERNS = [
        r"new\s+building",
        r"construction\s+of\s+a\s+new",
        r"constructing\s+a\s+new",
        r"erect(?:ing)?\s+a\s+new",
        r"propos(?:es|ing)\s+a\s+new\s+building",
        r"replace(?:s|d|ment)?\s+.*\s+with\s+a\s+new\s+building",
        # Additional patterns:
        r"construction\s+of\s+new",
        r"construction\s+of\s+a\s+new",
        r"construct(?:ing)?\s+.*\s+new\s+building",
        r"new\s+\d+-story\s+building",
        r"new\s+structure",
        r"propos(?:es|ing)\s+.*new\s+building",
    ]

    if new_imp is None:
        for pat in NEW_BUILDING_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                new_imp = 1  # assume some new impervious area
                break

    # Default to zero if nothing indicates new impervious area
    if new_imp is None:
        new_imp = 0

    return ParsedDescription(
        text=text,
        street_address=street_address,
        borough=borough,
        disturbed_area_sf=disturbed_area,
        new_impervious_sf=new_imp,
    )


# Helper to convert square footage strings into numbers
def _extract_sf_number(sf_string: str) -> float:
    """Convert strings like '12,000 SF' into float 12000."""
    cleaned = re.sub(r"[^\d]", "", sf_string)
    return float(cleaned) if cleaned else None