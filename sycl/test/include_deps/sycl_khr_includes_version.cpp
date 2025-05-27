// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/version | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/version>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/version
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-EMPTY:
