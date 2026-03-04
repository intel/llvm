// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/functional.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/functional.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/functional.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: functional.hpp
// CHECK-EMPTY:
