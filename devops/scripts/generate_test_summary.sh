#!/bin/bash
# shellcheck disable=SC2207

# Generate test summary from LIT output log
# Usage: generate_test_summary.sh <log_file> <test_type>
#   log_file: Path to the LIT output log file
#   test_type: Test category name (e.g., "Adapter Tests", "Conformance Tests")
#
# Security: Uses awk for text processing to avoid command injection
# Performance: Single-pass awk processing for efficiency

set -euo pipefail

LOG_FILE="${1:-}"
TEST_TYPE="${2:-Tests}"

if [ -z "$LOG_FILE" ]; then
  echo "Usage: $0 <log_file> <test_type>" >&2
  exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
  printf '### %s Summary\n\n' "$TEST_TYPE"
  printf '⚠️ Log file not found: %s\n\n' "$LOG_FILE"
  exit 0
fi

# Parse log file in single pass using awk for efficiency
# Uses 'sub' instead of 'gsub' for more precise regex matching
# Processes each line once and extracts test names by category
# Output format: STATUS:test_name for efficient array population (O(n) instead of O(n²))
parse_results() {
  awk '
    # Extract and clean test name from LIT status line
    function extract_test_name(prefix) {
      line = $0
      sub("^" prefix ": ", "", line)
      sub(/ \([0-9]+ of [0-9]+\)$/, "", line)
      return line
    }

    /^PASS: /        { print "PASS:" extract_test_name("PASS") }
    /^FAIL: /        { print "FAIL:" extract_test_name("FAIL") }
    /^XFAIL: /       { print "XFAIL:" extract_test_name("XFAIL") }
    /^XPASS: /       { print "XPASS:" extract_test_name("XPASS") }
    /^UNSUPPORTED: / { print "UNSUPPORTED:" extract_test_name("UNSUPPORTED") }
    /^TIMEOUT: /     { print "TIMEOUT:" extract_test_name("TIMEOUT") }
    /^UNRESOLVED: /  { print "UNRESOLVED:" extract_test_name("UNRESOLVED") }
    /^SKIPPED: /     { print "SKIPPED:" extract_test_name("SKIPPED") }
  ' "$LOG_FILE"
}

# Parse results with error handling
if ! results=$(parse_results); then
  printf "ERROR: Failed to parse test results from %s\n" "$LOG_FILE" >&2
  exit 1
fi

# Memory optimization: Use arrays instead of string concatenation (O(n) vs O(n²))
# Initialize arrays for each test status category
declare -a timeout_tests=()
declare -a fail_tests=()
declare -a xpass_tests=()
declare -a xfail_tests=()
declare -a unsup_tests=()
declare -a unres_tests=()
declare -a skip_tests=()
declare -a pass_tests=()

# Parse results and populate arrays efficiently
while IFS=: read -r status test_name; do
  case "$status" in
    TIMEOUT) timeout_tests+=("$test_name") ;;
    FAIL) fail_tests+=("$test_name") ;;
    XPASS) xpass_tests+=("$test_name") ;;
    XFAIL) xfail_tests+=("$test_name") ;;
    UNSUPPORTED) unsup_tests+=("$test_name") ;;
    UNRESOLVED) unres_tests+=("$test_name") ;;
    SKIPPED) skip_tests+=("$test_name") ;;
    PASS) pass_tests+=("$test_name") ;;
  esac
done <<< "$results"

# Calculate counts from array lengths
timeout_count=${#timeout_tests[@]}
fail_count=${#fail_tests[@]}
xpass_count=${#xpass_tests[@]}
xfail_count=${#xfail_tests[@]}
unsup_count=${#unsup_tests[@]}
unres_count=${#unres_tests[@]}
skip_count=${#skip_tests[@]}
pass_count=${#pass_tests[@]}

# Generate summary
printf '### %s Summary\n' "$TEST_TYPE"

# Print results (priority order: critical issues first)
# Critical issues (TIMEOUT, FAIL, XPASS) are open by default
# Other categories are collapsed to save space

# Helper function to print test list with indentation
print_test_list() {
  local -n tests_ref=$1
  for test in "${tests_ref[@]}"; do
    printf '  %s\n' "$test"
  done
}

# Define display order and labels
declare -a category_order=("TIMEOUT" "FAIL" "XPASS" "XFAIL" "UNSUPPORTED" "UNRESOLVED" "SKIPPED" "PASS")
declare -A category_labels=(
  ["TIMEOUT"]="Timeout Tests"
  ["FAIL"]="Failed Tests"
  ["XPASS"]="Unexpected Passed Tests (XPASS)"
  ["XFAIL"]="Expected Failures (XFAIL)"
  ["UNSUPPORTED"]="Unsupported Tests"
  ["UNRESOLVED"]="Unresolved Tests"
  ["SKIPPED"]="Skipped Tests"
  ["PASS"]="Passed Tests"
)
declare -A category_arrays=(
  ["TIMEOUT"]="timeout_tests"
  ["FAIL"]="fail_tests"
  ["XPASS"]="xpass_tests"
  ["XFAIL"]="xfail_tests"
  ["UNSUPPORTED"]="unsup_tests"
  ["UNRESOLVED"]="unres_tests"
  ["SKIPPED"]="skip_tests"
  ["PASS"]="pass_tests"
)
declare -a open_by_default=("TIMEOUT" "FAIL" "XPASS")

for cat in "${category_order[@]}"; do
  declare -n test_array="${category_arrays[$cat]}"
  count=${#test_array[@]}

  if [ "$count" -gt 0 ]; then
    # Check if this category should be open by default
    open_attr=""
    for open_cat in "${open_by_default[@]}"; do
      if [ "$cat" = "$open_cat" ]; then
        open_attr=" open"
        break
      fi
    done

    printf '\n<details%s>\n<summary>%s (%d)</summary>\n\n' "$open_attr" "${category_labels[$cat]}" "$count"
    print_test_list test_array
    printf '\n</details>\n'
  fi
done

# Total summary statistics
total=$((pass_count + fail_count + xfail_count + xpass_count + unsup_count + timeout_count + unres_count + skip_count))

printf '\n\n### Testing Time Summary:\n'
printf '  Total Discovered Tests: %d\n' "$total"
printf '  Passed           : %d\n' "$pass_count"
printf '  Failed           : %d\n' "$fail_count"
printf '  Expected Failed  : %d\n' "$xfail_count"
printf '  Unexpected Passed: %d\n' "$xpass_count"
printf '  Unsupported      : %d\n' "$unsup_count"
printf '  Timed Out        : %d\n' "$timeout_count"
printf '  Unresolved       : %d\n' "$unres_count"
printf '  Skipped          : %d\n' "$skip_count"
printf '\n'

# Exit with error if there are failures or timeouts
if [ "$fail_count" -gt 0 ] || [ "$timeout_count" -gt 0 ]; then
  exit 1
fi

exit 0
