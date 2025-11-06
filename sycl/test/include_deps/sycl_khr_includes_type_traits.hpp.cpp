// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/type_traits.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/type_traits.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/type_traits.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: device_aspect_traits.hpp
// CHECK-NEXT: device_aspect_macros.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-EMPTY:
