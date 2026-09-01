"""Parse statistics from LIT output."""

from typing import Dict, List

from ..constants import STAT_LINE_PATTERN


def parse_statistics(stats: List[str]) -> Dict[str, int]:
    result = {}
    for stat in stats:
        match = STAT_LINE_PATTERN.match(stat)
        if match:
            label = match.group(1)
            if label.endswith(" Tests"):
                label = label[: -len(" Tests")]
            result[label] = int(match.group(2))
    return result
