"""Parse statistics from LIT output."""
import re
from typing import List


def get_count_from_stats(stats: List[str], keywords: List[str]) -> int:
    """Extract count for specific test category from statistics.
    
    Args:
        stats: List of statistics lines from LIT output.
        keywords: Keywords to search for (e.g., ['Skipped', 'Unsupported']).
    
    Returns:
        Count of tests matching any of the keywords, or 0 if not found.
    """
    # Build regex pattern from keywords (compile once per call)
    pattern = re.compile("|".join(re.escape(kw) for kw in keywords))

    for stat in stats:
        if pattern.search(stat):
            match = re.search(r"(\d+)", stat)
            if match:
                return int(match.group(1))
    return 0
