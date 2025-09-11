// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/vec | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/vec>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/vec
// CHECK-NEXT: vector.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: detail/named_swizzles_mixin.hpp
// CHECK-NEXT: detail/vector_arith.hpp
// CHECK-EMPTY:
