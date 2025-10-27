// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/span.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/span.hpp>:
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: khr/includes/span.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-NEXT: sycl_span.hpp
// CHECK-EMPTY:
