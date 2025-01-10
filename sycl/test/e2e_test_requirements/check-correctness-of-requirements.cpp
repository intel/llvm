// This test checks that all "REQUIRES", "XFAIL" and "UNSUPPORTED" strings
// contain the right feature names.
//
// If this test fails:
// 1. there is some typo/deleted feature requested in the new/modified test.
// 2. ...or, there is some new feature. In this case please update the set of
//    features in sycl/test-e2e/sycl_lit_features.py.
//
// Get a set of all features passed to "REQUIRES", "XFAIL" and "UNSUPPORTED":
// RUN: llvm-lit --show-used-features %S/../../test-e2e > %t
//
// Process this set using a python script:
// RUN: python3 %S/check-correctness-of-requirements.py %t
//
// Small negative test
// RUN: echo non-existing-feature-request > %t | not python3 %S/check-correctness-of-requirements.py %t
