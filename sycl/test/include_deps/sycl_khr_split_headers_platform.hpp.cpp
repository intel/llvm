// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/split_headers/platform.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/split_headers/platform.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/split_headers/platform.hpp
// CHECK-NEXT: khr/split_headers/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: platform.hpp
// CHECK-NEXT: backend_types.hpp
// CHECK-NEXT: detail/abi_neutral.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/info_desc_helpers.hpp
// CHECK-NEXT: detail/info_desc_traits.hpp
// CHECK-NEXT: ur_api.h
// CHECK-NEXT: info/info_desc.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: detail/device_info_types.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/bool_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: ext/codeplay/experimental/max_registers_query.hpp
// CHECK-NEXT: ext/intel/info/device.hpp
// CHECK-NEXT: ext/intel/info/kernel.hpp
// CHECK-NEXT: ext/oneapi/experimental/bindless_image_info.hpp
// CHECK-NEXT: ext/oneapi/experimental/composite_device.hpp
// CHECK-NEXT: ext/oneapi/experimental/device_architecture.hpp
// CHECK-NEXT: ext/oneapi/experimental/device_architecture.def
// CHECK-NEXT: ext/oneapi/experimental/forward_progress.hpp
// CHECK-NEXT: ext/oneapi/experimental/kernel_queue_info.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: ext/oneapi/experimental/max_work_groups.hpp
// CHECK-NEXT: ext/oneapi/info/device.hpp
// CHECK-NEXT: ext/oneapi/matrix/query-types.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: detail/half_type_impl.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: ext/oneapi/matrix/matrix-unified-utils.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: detail/string_view.hpp
// CHECK-NEXT: device_selector.hpp
// CHECK-EMPTY:
