// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/detail/export.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/detail/export.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: detail/export.hpp
// CHECK-EMPTY:
