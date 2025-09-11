// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/span | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/span>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/span
// CHECK-NEXT: sycl_span.hpp
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-EMPTY:
