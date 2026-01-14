// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/kernel_handler.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/kernel_handler.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/kernel_handler.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: kernel_handler.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-EMPTY:
