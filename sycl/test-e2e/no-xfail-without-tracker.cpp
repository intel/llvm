// This test is intended to ensure that we have no trackers marked as XFAIL
// without a tracker information added to a test.
//
// The format we check is:
// XFAIL: lit,features
// XFAIL-TRACKER: [GitHub issue URL|Internal tracker ID]
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
// - search for all "XFAIL" occurrences, display line with match and the next one
//   -I, --include to drop binary files and other unrelated files
// - in the result, search for "XFAIL" again, but invert the result - this
//   allows us to get the line *after* XFAIL
// - in those lines, check that XFAIL-TRACKER is present and correct. Once
//   again, invert the search to get all "bad" lines
// - make a final count of how many ill-formatted directives there are and
//   verify that against the reference
//
// RUN: grep -rI "XFAIL:" %S -A 1 --include=*.c --include=*.cpp \
// RUN:     --no-group-separator | \
// RUN: grep -v "XFAIL:" | \
// RUN: grep -Pv "XFAIL-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" | \
// RUN: wc -l | FileCheck %s --check-prefix NUMBER-OF-XFAIL-WITHOUT-TRACKER
//
// The number below is a number of tests which are *improperly* XFAIL-ed, i.e.
// we either don't have a tracker associated with a failure listed in those
// tests, or it is listed in a wrong format.
// Note: strictly speaking, that is not amount of files, but amount of XFAIL
// directives. If a test contains several XFAIL directives, some of them may be
// valid and other may not.
//
// That number *must not* increase. Any PR which causes this number to grow
// should be rejected and it should be updated to either keep the number as-is
// or have it reduced (preferrably, down to zero).
//
// If you see this test failed for your patch, it means that you either
// introduced XFAIL directive to a test improperly, or broke the format of an
// existing XFAIL-ed tests.
// Another possibility (and that is a good option) is that you updated some
// tests to match the required format and in that case you should just update
// (i.e. reduce) the number below.
//
// NUMBER-OF-XFAIL-WITHOUT-TRACKER: 178
