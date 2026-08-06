"""Test data validation."""
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.test_data import TestLists


def validate_test_counts(
    total_discovered: int,
    test_lists: "TestLists",
    displayed_skipped: int,
    displayed_excluded: int,
) -> None:
    """Validate test counts match (prints warnings on mismatch)."""
    if not total_discovered:
        return

    sum_categories = sum(len(tests) for tests in test_lists.values())

    # Add skipped tests if they were displayed but not in test_lists
    if displayed_skipped > 0 and "Skipped" not in test_lists:
        sum_categories += displayed_skipped

    # Add excluded tests if they were displayed but not in test_lists
    if displayed_excluded > 0 and "Excluded" not in test_lists:
        sum_categories += displayed_excluded

    if total_discovered != sum_categories:
        print()
        print(
            f"::warning::Test count mismatch: Total Discovered = "
            f"{total_discovered}, but sum of all categories = {sum_categories}"
        )
        print()
