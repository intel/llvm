// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/platform.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/platform.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/platform.hpp
// CHECK-NEXT: khr/includes/version.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: feature_test.hpp
// CHECK-NEXT: platform.hpp
// CHECK-NEXT: backend_types.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/info_desc_helpers.hpp
// CHECK-NEXT: aspects.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: info/aspects.def
// CHECK-NEXT: info/aspects_deprecated.def
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: info/info_desc.hpp
// CHECK-NEXT: ur_api.h
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: ext/oneapi/experimental/device_architecture.hpp
// CHECK-NEXT: ext/oneapi/experimental/device_architecture.def
// CHECK-NEXT: ext/oneapi/experimental/forward_progress.hpp
// CHECK-NEXT: ext/oneapi/matrix/query-types.hpp
// CHECK-NEXT: ext/oneapi/bfloat16.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: half_type.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: ext/oneapi/matrix/matrix-unified-utils.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: info/platform_traits.def
// CHECK-NEXT: info/context_traits.def
// CHECK-NEXT: info/device_traits_2020_deprecated.def
// CHECK-NEXT: info/device_traits_deprecated.def
// CHECK-NEXT: info/device_traits.def
// CHECK-NEXT: info/queue_traits.def
// CHECK-NEXT: info/kernel_traits.def
// CHECK-NEXT: info/kernel_device_specific_traits.def
// CHECK-NEXT: info/event_traits.def
// CHECK-NEXT: info/event_profiling_traits.def
// CHECK-NEXT: info/ext_codeplay_device_traits.def
// CHECK-NEXT: info/ext_intel_device_traits.def
// CHECK-NEXT: info/ext_intel_kernel_info_traits.def
// CHECK-NEXT: info/ext_oneapi_device_traits.def
// CHECK-NEXT: info/ext_oneapi_kernel_queue_specific_traits.def
// CHECK-NEXT: info/sycl_backend_traits.def
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: detail/string_view.hpp
// CHECK-NEXT: detail/util.hpp
// CHECK-NEXT: device_selector.hpp
// CHECK-EMPTY:
