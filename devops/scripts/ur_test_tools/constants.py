"""Constants for UR test tools."""

import re

# File I/O
# Conservative upper bound for locating the LIT "Testing:" marker.
# Non-verbose CI runs normally emit it much earlier.
MAX_LINES_TO_SCAN = 1000

SEPARATOR_WIDTH = 70

# LIT Configuration
DEFAULT_LIT_TIMEOUT = 120
DEFAULT_LIT_JOBS = 50

# Test Type Identifiers
TEST_TYPE_ADAPTER_SPECIFIC = "adapter-specific"
TEST_TYPE_CONFORMANCE = "conformance"

# LIT Test Filters
# These tests cause timeouts on CI and are excluded from adapter-specific runs
LIT_FILTER_OUT_ADAPTER_SPECIFIC = (
    "(adapters/level_zero/memcheck.test|"
    "adapters/level_zero/v2/deferred_kernel_memcheck.test)"
)

# Constants
TEST_NOT_SELECTED_MSG = "Test not selected"
SLOWEST_TESTS_HEADER = "Slowest Tests:"
TEST_TIMES_HEADERS = ("Tests Times:", "Test Times:")

# LIT Output Patterns
FAIL_TIMEOUT_PATTERN = re.compile(r"^(FAIL|TIMEOUT):")
TEST_LIST_HEADER_PATTERN = re.compile(
    r"^(Passed|Unsupported|Failed|Expectedly Failed|"
    r"Timed Out|Unexpectedly Passed|Unresolved) Tests \("
)
STATS_PATTERN = re.compile(
    r"^\s*(Total Discovered|Expected Passes|Expectedly Failed|"
    r"Excluded|Unsupported|Skipped|Passed|Passed With Retry|"
    r"Failed|Timed Out|Unexpectedly Passed|Unresolved)(\s+Tests)?\s*:"
)
TEST_CATEGORY_PATTERN = re.compile(r"^([A-Za-z]+(?: [A-Za-z]+)*) Tests \((\d+)\):")
