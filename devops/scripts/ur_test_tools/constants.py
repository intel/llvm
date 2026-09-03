"""Constants for UR test tools."""

import re

# Conservative upper bound for locating the LIT "Testing:" marker.
# Non-verbose CI runs normally emit it much earlier.
MAX_LINES_TO_SCAN = 1000

SEPARATOR_WIDTH = 70

# LIT_COMMON_REPORTING_OPTIONS: what to show, kept aligned with standalone
# builds (unified-runtime/test/CMakeLists.txt).
# LIT_CI_OPTIONS: how to format output for CI tooling; standalone uses
# --succinct instead.
LIT_COMMON_REPORTING_OPTIONS = [
    "--show-unsupported",
    "--show-pass",
    "--show-xfail",
    "--time-tests",
    "--show-flakypass",
    "--show-skipped",
    "--show-excluded",
]

LIT_CI_OPTIONS = [
    "-v",
    "--no-progress-bar",
]

DEFAULT_LIT_TIMEOUT = 120
DEFAULT_LIT_JOBS = 50

TEST_TYPE_ADAPTER_SPECIFIC = "adapter-specific"
TEST_TYPE_CONFORMANCE = "conformance"

# These tests cause timeouts on CI and are excluded from adapter-specific runs
LIT_FILTER_OUT_ADAPTER_SPECIFIC = (
    "(adapters/level_zero/memcheck.test|"
    "adapters/level_zero/v2/deferred_kernel_memcheck.test)"
)

TEST_NOT_SELECTED_MSG = "Test not selected"
SLOWEST_TESTS_HEADER = "Slowest Tests:"
TEST_TIMES_HEADERS = ("Tests Times:", "Test Times:")

FAIL_TIMEOUT_PATTERN = re.compile(r"^(FAIL|TIMEOUT):")
TEST_CATEGORY_PATTERN = re.compile(r"^([A-Za-z]+(?: [A-Za-z]+)*) Tests \((\d+)\):")
TESTING_TIME_PATTERN = re.compile(r"^\s*Testing Time:\s*([0-9.]+)s")
STAT_LINE_PATTERN = re.compile(r"^\s*(.+?)\s*:\s*(\d+)(?:\s|$)")
