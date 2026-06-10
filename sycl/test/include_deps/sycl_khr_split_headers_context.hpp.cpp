// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/context.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/context.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/context.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: context.hpp
// CHECK-NEXT: async_handler.hpp
// CHECK-NEXT: backend_types.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: info/context.hpp
// CHECK-NEXT: detail/info_desc_traits.hpp
// CHECK-NEXT: ur_api.h
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: property_list.hpp
// CHECK-NEXT: detail/property_helper.hpp
// CHECK-NEXT: detail/property_list_base.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: properties/property_traits.hpp
// CHECK-NEXT: usm/usm_enums.hpp
// CHECK-EMPTY:
