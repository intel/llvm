// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/index_space.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/index_space.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/index_space.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: item.hpp
// CHECK-NEXT: detail/item_base.hpp
// CHECK-NEXT: nd_item.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: device_event.hpp
// CHECK-NEXT: group.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: pointers.hpp
// CHECK-NEXT: nd_range.hpp
// CHECK-EMPTY:
