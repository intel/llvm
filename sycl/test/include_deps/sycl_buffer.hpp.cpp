// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/buffer.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/buffer.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: buffer.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: CL/__spirv/spirv_ops.hpp
// CHECK-NEXT: CL/__spirv/spirv_types.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: backend_types.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: detail/pi.h
// CHECK-NEXT: detail/pi_error.def
// CHECK-NEXT: detail/pi.def
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/cl.h
// CHECK-NEXT: CL/cl.h
// CHECK-NEXT: CL/cl_version.h
// CHECK-NEXT: CL/cl_platform.h
// CHECK-NEXT: CL/cl_ext.h
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/helpers.hpp
// CHECK-NEXT: detail/pi.hpp
// CHECK-NEXT: detail/os_util.hpp
// CHECK-NEXT: memory_enums.hpp
// CHECK-NEXT: CL/__spirv/spirv_vars.hpp
// CHECK-NEXT: detail/iostream_proxy.hpp
// CHECK-NEXT: detail/is_device_copyable.hpp
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: detail/property_helper.hpp
// CHECK-NEXT: detail/stl_type_traits.hpp
// CHECK-NEXT: detail/sycl_mem_obj_allocator.hpp
// CHECK-NEXT: detail/aligned_allocator.hpp
// CHECK-NEXT: ext/oneapi/accessor_property_list.hpp
// CHECK-NEXT: detail/property_list_base.hpp
// CHECK-NEXT: property_list.hpp
// CHECK-NEXT: properties/property_traits.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: range.hpp
// CHECK-EMPTY:
