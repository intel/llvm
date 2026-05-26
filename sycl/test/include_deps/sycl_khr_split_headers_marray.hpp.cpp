// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/marray.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/marray.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/marray.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/assert.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/nd_loop.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-EMPTY:
