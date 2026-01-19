// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/atomic.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/atomic.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/atomic.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: atomic_fence.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: detail/spirv.hpp
// CHECK-NEXT: __spirv/spirv_ops.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: detail/fwd/accessor.hpp
// CHECK-NEXT: detail/memcpy.hpp
// CHECK-NEXT: atomic_ref.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: ext/oneapi/experimental/address_cast.hpp
// CHECK-EMPTY:
