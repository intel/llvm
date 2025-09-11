// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/atomic | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/atomic>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/atomic
// CHECK-NEXT: atomic_ref.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: ext/oneapi/experimental/address_cast.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: detail/spirv.hpp
// CHECK-NEXT: __spirv/spirv_ops.hpp
// CHECK-NEXT: ext/oneapi/experimental/non_uniform_groups.hpp
// CHECK-NEXT: ext/oneapi/sub_group_mask.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: range.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: detail/is_device_copyable.hpp
// CHECK-NEXT: sub_group.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: nd_item.hpp
// CHECK-NEXT: device_event.hpp
// CHECK-NEXT: group.hpp
// CHECK-NEXT: detail/item_base.hpp
// CHECK-NEXT: h_item.hpp
// CHECK-NEXT: item.hpp
// CHECK-NEXT: pointers.hpp
// CHECK-NEXT: nd_range.hpp
// CHECK-NEXT: vector.hpp
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_arith.hpp
// CHECK-NEXT: atomic_fence.hpp
// CHECK-EMPTY:
