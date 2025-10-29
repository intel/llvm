// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/math.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/math.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/math.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: builtins.hpp
// CHECK-NEXT: detail/builtins/builtins.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/vector_convert.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: vector.hpp
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_arith.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/fwd/accessor.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: detail/builtins/common_functions.inc
// CHECK-NEXT: detail/builtins/helper_macros.hpp
// CHECK-NEXT: detail/builtins/geometric_functions.inc
// CHECK-NEXT: detail/builtins/half_precision_math_functions.inc
// CHECK-NEXT: detail/builtins/integer_functions.inc
// CHECK-NEXT: detail/builtins/math_functions.inc
// CHECK-NEXT: detail/builtins/native_math_functions.inc
// CHECK-NEXT: detail/builtins/relational_functions.inc
// CHECK-EMPTY:
