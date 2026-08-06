"""Constants for UR test tools."""
import re

# File I/O
MAX_LINES_TO_SCAN = 1000
# Rationale: LIT outputs "-- Testing: N tests, M workers --" at the START of run,
# typically within first 50 lines. 1000 lines provides safety margin for cmake
# build output that may precede test execution.

SEPARATOR_WIDTH = 70

# Job Calculation
MAX_JOBS = 16
# Rationale: Controls `cmake --build -j N` parallelism, NOT test execution
# (LIT uses separate `-j 50` in LIT_OPTS). Prevents resource exhaustion:
#   - Memory: Each cmake job can use ~500MB-1GB during compilation/linking
#   - I/O: Too many parallel builds can saturate disk on shared CI runners
#   - On 96-core machines: nproc/3 = 32, capped to 16 to stay under ~16GB peak

# LIT Configuration
DEFAULT_LIT_TIMEOUT = 120
DEFAULT_LIT_JOBS = 50

# Test Type Identifiers
TEST_TYPE_ADAPTER_SPECIFIC = "adapter-specific"
TEST_TYPE_UNIT = "unit"

# Display Strings
TEST_NOT_SELECTED_MSG = "Test not selected"
SLOWEST_TESTS_HEADER = "Slowest Tests:"
TEST_TIMES_HEADERS = ("Tests Times:", "Test Times:")

# LIT Output Patterns
# These patterns parse text output from LLVM LIT (llvm-lit).
# Tested with: LLVM 15+ (stable format since ~2010)
# LIT flags used: --verbose --time-tests --show-unsupported --show-pass --show-xfail
# If LIT changes output format in future versions, update these patterns accordingly.
# For structured data (test counts, skipped tests), prefer using --xunit-xml-output.

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
