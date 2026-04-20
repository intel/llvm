// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/builtins_math.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/builtins_math.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/builtins_math.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: builtins_math.hpp
// CHECK-NEXT: detail/builtins/half_precision_math_functions.hpp
// CHECK-NEXT: detail/builtins/builtin_helpers.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/half_type_impl.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: detail/loop.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/bool_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/vector_core.hpp
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/assert.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/nd_loop.hpp
// CHECK-NEXT: detail/fwd/accessor.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: detail/type_traits/integer_traits.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: detail/builtins/helper_macros.hpp
// CHECK-NEXT: detail/builtins/math_functions.hpp
// CHECK-NEXT: detail/builtins/native_math_functions.hpp
// CHECK-EMPTY:
