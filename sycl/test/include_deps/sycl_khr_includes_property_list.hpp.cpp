// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/property_list.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/property_list.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/property_list.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: property_list.hpp
// CHECK-NEXT: detail/property_helper.hpp
// CHECK-NEXT: detail/property_list_base.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: properties/property_traits.hpp
// CHECK-EMPTY:
