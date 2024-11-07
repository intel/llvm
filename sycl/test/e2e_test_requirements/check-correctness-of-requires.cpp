// This test checks that all "REQUIRES" strings contain the right feature names
// If this test fails:
// 1. there is some typo/non-existing feature request in the
//    modified test.
// 2. ...or, there is some new feature. In this case please update the set of
//    features in check-correctness-of-requires.py
//
// RUN: grep -rI --include=*.cpp "REQUIRES: " %S/../../test-e2e | sed -E 's|.*/test-e2e/||' > %t
// Using a python script as it's easier to work with sets there
// RUN: python3 %S/check-correctness-of-requires.py %t %sycl_include
