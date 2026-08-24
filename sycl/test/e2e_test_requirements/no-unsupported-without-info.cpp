// This test verifies that there are no untracked UNSUPPORTED tests.

// If this test fails for your patch, it means that you either introduced
// an UNSUPPORTED directive incorrectly, or broke the format of an
// existing UNSUPPORTED test.

// For more info see:
// https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/README.md#marking-tests-as-unsupported

// The expected format is:
// UNSUPPORTED: lit,features
// UNSUPPORTED-TRACKER: [GitHub issue URL|Internal tracker ID]
// *OR*
// UNSUPPORTED: lit,features
// UNSUPPORTED-INTENDED: explanation why the test isn't intended to run
// with this feature

// GitHub issue URL format:
//     https://github.com/owner/repo/issues/12345

// Internal tracker ID format:
//     PROJECT-123456

// REQUIRES: linux

// Command explanation:
// - Search for all "UNSUPPORTED" occurrences and print each matching line
//   plus the next line. The -I and --include options skip binary and
//   unrelated files.
// - In that output, search for "UNSUPPORTED" again and invert the match.
//   This leaves the line *after* each UNSUPPORTED line.
// - In those lines, verify that UNSUPPORTED-TRACKER or
//   UNSUPPORTED-INTENDED is present and correctly formatted. Invert this
//   match again to keep only "bad" lines.
// - There must be no bad lines, so we assert that the final grep fails
//   by using the not operator.

// RUN: grep -rI "UNSUPPORTED:" %S/../../test-e2e -A 1 --include=*.cpp --no-group-separator | \
// RUN: grep -v "UNSUPPORTED:" | \
// RUN: not grep -Pv "(?:UNSUPPORTED-TRACKER:\s+(?:(?:https:\/\/github.com\/[\w\d-]+\/[\w\d-]+\/issues\/[\d]+)|(?:[\w]+-[\d]+)))|(?:UNSUPPORTED-INTENDED:\s*.+)" > %t
