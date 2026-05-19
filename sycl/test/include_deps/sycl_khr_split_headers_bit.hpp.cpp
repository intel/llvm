// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/bit.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/bit.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/bit.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-EMPTY:
