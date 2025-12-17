// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/vec.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/vec.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/vec.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: vector.hpp
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_arith.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/fwd/accessor.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-EMPTY:
