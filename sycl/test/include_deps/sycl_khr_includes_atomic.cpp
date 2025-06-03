// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/atomic | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/atomic>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/atomic
// CHECK-NEXT: atomic_ref.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: __spirv/spirv_ops.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: ext/oneapi/experimental/address_cast.hpp
// CHECK-NEXT: detail/spirv.hpp
// CHECK-NEXT: ext/oneapi/experimental/non_uniform_groups.hpp
// CHECK-NEXT: ext/oneapi/sub_group_mask.hpp
// CHECK-NEXT: builtins.hpp
// CHECK-NEXT: detail/builtins/builtins.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/vector_convert.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: vector.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_arith.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: detail/is_device_copyable.hpp
// CHECK-NEXT: detail/builtins/common_functions.inc
// CHECK-NEXT: detail/builtins/helper_macros.hpp
// CHECK-NEXT: detail/builtins/geometric_functions.inc
// CHECK-NEXT: detail/builtins/half_precision_math_functions.inc
// CHECK-NEXT: detail/builtins/integer_functions.inc
// CHECK-NEXT: detail/builtins/math_functions.inc
// CHECK-NEXT: detail/builtins/native_math_functions.inc
// CHECK-NEXT: detail/builtins/relational_functions.inc
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: atomic_fence.hpp
// CHECK-EMPTY:
