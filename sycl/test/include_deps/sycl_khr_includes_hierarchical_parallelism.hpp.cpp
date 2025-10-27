// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/hierarchical_parallelism.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/hierarchical_parallelism.hpp>:
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: detail/item_base.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: device_event.hpp
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: group.hpp
// CHECK-NEXT: h_item.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: item.hpp
// CHECK-NEXT: khr/includes/hierarchical_parallelism.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: pointers.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-EMPTY:
