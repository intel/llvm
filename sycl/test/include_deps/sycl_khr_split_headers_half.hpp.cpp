// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/half.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/half.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/half.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: detail/half_type_impl.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-EMPTY:
