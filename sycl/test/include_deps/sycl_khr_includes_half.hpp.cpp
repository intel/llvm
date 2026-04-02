// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/half.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/half.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/half.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-EMPTY:
