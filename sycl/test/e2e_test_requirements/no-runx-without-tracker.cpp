// This test is intended to ensure that we have no tests marked as RUNx
// without a tracker information added to a test. If it fails - please create
// a tracker for RUNx-marked test.
//
// The format we check is:
// RUNx: command
// RUNx-TRACKER: [GitHub issue URL|Internal tracker ID]
//
// GitHub issue URL format:
//     https://github.com/owner/repo/issues/12345
//
// Internal tracker ID format:
//     PROJECT-123456
//
// REQUIRES: linux
//
// Explanation of the command:
// - search for all "RUNx" occurrences, display line with match and the next one
//   -I, --include to drop binary files and other unrelated files
// - in the result, search for "RUNx" again, but invert the result - this
//   allows us to get the line *after* RUNx
// - in those lines, check that RUNx-TRACKER is present and correct. Once
//   again, invert the search to get all "bad" lines; running it with "not" as
//   grep exits with 1 if it finds nothing
//
// RUN: grep -rI "RUNx:" %S/../../test-e2e \
// RUN: -A 1 --include=*.cpp --no-group-separator | \
// RUN: grep -v "RUNx:" | \
// RUN: not grep -Pv "RUNx-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" 
