// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/detail/defines_elementary.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/detail/defines_elementary.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-EMPTY:
