#!/bin/bash

# Generate test summary from LIT output log
# Usage: generate_test_summary.sh <log_file> <test_type>

set -euo pipefail

LOG_FILE="${1:-}"
TEST_TYPE="${2:-Tests}"

if [ -z "$LOG_FILE" ]; then
  echo "Usage: $0 <log_file> <test_type>" >&2
  exit 1
fi

# Basic input validation
if [[ "$LOG_FILE" =~ [\;\&\|\`\$] ]]; then
  echo "ERROR: Invalid characters in log file path" >&2
  exit 1
fi

# Sanitize TEST_TYPE to prevent HTML/markdown injection
# Allow only alphanumeric, spaces, hyphens, underscores
TEST_TYPE=$(echo "$TEST_TYPE" | tr -cd '[:alnum:][:space:]-_' | tr -s ' ')

if [ ! -f "$LOG_FILE" ]; then
  printf '### %s Summary\n\n' "$TEST_TYPE"
  printf 'Log file not found: %s\n\n' "$LOG_FILE"
  exit 0
fi

# Check if file is empty
if [ ! -s "$LOG_FILE" ]; then
  printf '### %s Summary\n\n' "$TEST_TYPE"
  printf 'Log file is empty: %s\n\n' "$LOG_FILE"
  exit 0
fi

# Parse LIT summary statistics
parse_summary_stats() {
  awk '
    /^Total Discovered Tests:/ { gsub(/[^0-9]/, ""); if ($0 != "") print "TOTAL:" $0 }
    /^  Skipped:/               { match($0, /[0-9]+/); if (RSTART > 0) print "SKIPPED_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Passed[ :]/             { match($0, /[0-9]+/); if (RSTART > 0) print "PASSED_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Failed[ :]/             { match($0, /[0-9]+/); if (RSTART > 0) print "FAILED_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Unsupported[ :]/        { match($0, /[0-9]+/); if (RSTART > 0) print "UNSUPPORTED_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Unresolved[ :]/         { match($0, /[0-9]+/); if (RSTART > 0) print "UNRESOLVED_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Timeout[ :]/            { match($0, /[0-9]+/); if (RSTART > 0) print "TIMEOUT_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Expected Passes[ :]/    { match($0, /[0-9]+/); if (RSTART > 0) print "XFAIL_COUNT:" substr($0, RSTART, RLENGTH) }
    /^  Unexpected Passes[ :]/  { match($0, /[0-9]+/); if (RSTART > 0) print "XPASS_COUNT:" substr($0, RSTART, RLENGTH) }
  ' "$LOG_FILE"
}

# Parse test results from log
parse_results() {
  awk '
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
  printf 'ERROR: Failed to parse test results from %s\n' "$LOG_FILE" >&2
  exit 1
fi

# Initialize arrays for test results
declare -a timeout_tests=()
declare -a fail_tests=()
declare -a xpass_tests=()
declare -a xfail_tests=()
declare -a unsup_tests=()
declare -a unres_tests=()
declare -a skip_tests=()
declare -a pass_tests=()

# Populate arrays from parsed results
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

# Parse summary statistics from LIT output (source of truth for counts)
# These override array-based counts where available, especially for skipped tests
# which don't appear in verbose output
summary_stats=$(parse_summary_stats)
total_tests=""
summary_skip_count=""
summary_pass_count=""
summary_fail_count=""
summary_unsup_count=""
summary_unres_count=""
summary_timeout_count=""
summary_xfail_count=""
summary_xpass_count=""

while IFS=: read -r key value; do
  case "$key" in
    TOTAL) total_tests="$value" ;;
    SKIPPED_COUNT) summary_skip_count="$value" ;;
    PASSED_COUNT) summary_pass_count="$value" ;;
    FAILED_COUNT) summary_fail_count="$value" ;;
    UNSUPPORTED_COUNT) summary_unsup_count="$value" ;;
    UNRESOLVED_COUNT) summary_unres_count="$value" ;;
    TIMEOUT_COUNT) summary_timeout_count="$value" ;;
    XFAIL_COUNT) summary_xfail_count="$value" ;;
    XPASS_COUNT) summary_xpass_count="$value" ;;
  esac
done <<< "$summary_stats"

# Calculate counts from array lengths
timeout_count=${#timeout_tests[@]}
fail_count=${#fail_tests[@]}
xpass_count=${#xpass_tests[@]}
xfail_count=${#xfail_tests[@]}
unsup_count=${#unsup_tests[@]}
unres_count=${#unres_tests[@]}
skip_count=${#skip_tests[@]}
pass_count=${#pass_tests[@]}

# Use summary counts where available (more accurate, especially for skipped tests)
[ -n "$summary_timeout_count" ] && timeout_count=$summary_timeout_count
[ -n "$summary_fail_count" ] && fail_count=$summary_fail_count
[ -n "$summary_xpass_count" ] && xpass_count=$summary_xpass_count
[ -n "$summary_xfail_count" ] && xfail_count=$summary_xfail_count
[ -n "$summary_unsup_count" ] && unsup_count=$summary_unsup_count
[ -n "$summary_unres_count" ] && unres_count=$summary_unres_count
[ -n "$summary_skip_count" ] && skip_count=$summary_skip_count
[ -n "$summary_pass_count" ] && pass_count=$summary_pass_count

# Generate summary
printf '### %s Summary\n' "$TEST_TYPE"

# Show overall statistics
if [ -n "$total_tests" ] && [ "$total_tests" -gt 0 ]; then
  printf '\n**Total Discovered Tests:** %s\n' "$total_tests"
fi
if [ "$pass_count" -gt 0 ]; then
  printf '**Passed:** %d\n' "$pass_count"
fi
if [ "$skip_count" -gt 0 ]; then
  printf '**Skipped:** %d\n' "$skip_count"
fi
if [ "$fail_count" -gt 0 ]; then
  printf '**Failed:** %d\n' "$fail_count"
fi
if [ "$xfail_count" -gt 0 ]; then
  printf '**Expected Failures:** %d\n' "$xfail_count"
fi
if [ "$xpass_count" -gt 0 ]; then
  printf '**Unexpected Passes:** %d\n' "$xpass_count"
fi
if [ "$unsup_count" -gt 0 ]; then
  printf '**Unsupported:** %d\n' "$unsup_count"
fi
if [ "$timeout_count" -gt 0 ]; then
  printf '**Timeouts:** %d\n' "$timeout_count"
fi
if [ "$unres_count" -gt 0 ]; then
  printf '**Unresolved:** %d\n' "$unres_count"
fi
printf '\n'

# Detailed test lists by category (collapsed)

# Helper function to print test list in code block
print_test_list() {
  local -n tests_ref=$1
  printf '\n```\n'
  for test in "${tests_ref[@]}"; do
    printf '%s\n' "$test"
  done
  printf '```\n'
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

# Exit with error if there are failures or timeouts
if [ "$fail_count" -gt 0 ] || [ "$timeout_count" -gt 0 ]; then
  exit 1
fi

exit 0
