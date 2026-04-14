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
// CHECK-NEXT: detail/async_work_group_copy_ptr.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/type_traits/integer_traits.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: detail/builder.hpp
// CHECK-NEXT: detail/spirv_memory_semantics.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/bool_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: device_event.hpp
// CHECK-NEXT: group.hpp
// CHECK-NEXT: detail/assert.hpp
// CHECK-NEXT: detail/nd_loop.hpp
// CHECK-NEXT: pointers.hpp
// CHECK-NEXT: nd_range.hpp
// CHECK-NEXT: detail/sub_group_core.hpp
// CHECK-EMPTY:
