// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/multi_ptr | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/multi_ptr>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/multi_ptr
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-EMPTY:
