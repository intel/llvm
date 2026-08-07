"""Parse statistics from LIT output."""

import re
from typing import List


def get_count_from_stats(stats: List[str], keywords: List[str]) -> int:
    # Build regex pattern from keywords (compile once per call)
    pattern = re.compile("|".join(re.escape(kw) for kw in keywords))

    for stat in stats:
        if pattern.search(stat):
            match = re.search(r"(\d+)", stat)
            if match:
                return int(match.group(1))
    return 0
