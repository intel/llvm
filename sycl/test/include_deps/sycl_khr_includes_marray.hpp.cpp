// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/marray.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/marray.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/marray.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-EMPTY:
