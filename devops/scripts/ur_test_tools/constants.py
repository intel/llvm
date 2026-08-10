"""Constants for UR test tools."""

import re

# File I/O
MAX_LINES_TO_SCAN = 1000

SEPARATOR_WIDTH = 70

# Job Calculation
MAX_JOBS = 16

# LIT Configuration
DEFAULT_LIT_TIMEOUT = 120
DEFAULT_LIT_JOBS = 50

# Test Type Identifiers
TEST_TYPE_ADAPTER_SPECIFIC = "adapter-specific"

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
