// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/event.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/event.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/event.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: event.hpp
// CHECK-NEXT: backend_types.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: info/event.hpp
// CHECK-NEXT: detail/info_desc_traits.hpp
// CHECK-NEXT: ur_api.h
// CHECK-EMPTY:
