// This test is intended to ensure that we have no tests marked as XFAIL
// without a tracker information added to a test.
// For more info see: sycl/test-e2e/README.md
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
//   again, invert the search to get all "bad" lines and save the test names in
//   the temp file
// - make a final count of how many ill-formatted directives there are and
//   verify that against the reference
// - ...and check if the list of improperly XFAIL-ed tests needs to be updated.
//
// RUN: grep -rI "XFAIL:" %S/../../test-e2e \
// RUN: -A 1 --include=*.cpp --no-group-separator | \
// RUN: grep -v "XFAIL:" | \
// RUN: grep -Pv "XFAIL-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" > %t
// RUN: cat %t | wc -l | FileCheck %s --check-prefix NUMBER-OF-XFAIL-WITHOUT-TRACKER
// RUN: cat %t | sed 's/\.cpp.*/.cpp/' | sort | wc - l | [ $(cat)  -gt 0 ] && FileCheck %s
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
// or have it reduced (preferably, down to zero).
//
// If you see this test failed for your patch, it means that you either
// introduced XFAIL directive to a test improperly, or broke the format of an
// existing XFAIL-ed tests.
// Another possibility (and that is a good option) is that you updated some
// tests to match the required format and in that case you should just update
// (i.e. reduce) the number and the list below.
//
// NUMBER-OF-XFAIL-WITHOUT-TRACKER: 0
//
// List of improperly XFAIL-ed tests.
// As an example, if test test-e2e/Foo/foo.cpp is improperly XFAIL-ed,
// add this line at the end of this file: // CHECK: Foo/foo.cpp
// or // CHECK-NEXT: Foo/foo.cpp in case the list is not empty.
// Remove the CHECK once the test has been properly XFAIL-ed.
