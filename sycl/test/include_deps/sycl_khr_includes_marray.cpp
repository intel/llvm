// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/marray | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/marray>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/marray
// CHECK-NEXT: marray.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: stl_wrappers/cstdlib
// CHECK-NEXT: stl_wrappers/cassert
// CHECK-NEXT: stl_wrappers/assert.h
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/is_device_copyable.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-EMPTY:
