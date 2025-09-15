// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/byte.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/byte.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/byte.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-EMPTY:
