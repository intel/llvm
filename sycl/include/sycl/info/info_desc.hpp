//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <unified-runtime/ur_api.h>

// FIXME: .def files included to this file use all sorts of SYCL objects like
// id, range, traits, etc. We have to include some headers before including .def
// files.
#include <sycl/aspects.hpp>
#include <sycl/detail/device_info_types.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/codeplay/experimental/max_registers_query.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/intel/info/kernel.hpp>
#include <sycl/ext/oneapi/experimental/bindless_image_info.hpp>
#include <sycl/ext/oneapi/experimental/composite_device.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/experimental/kernel_queue_info.hpp>
#include <sycl/ext/oneapi/experimental/max_work_groups.hpp>
#include <sycl/ext/oneapi/info/device.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>

#include <sycl/range.hpp>

#include <string>
#include <vector>

// This is used in trait .def files when there isn't a corresponding backend
// query but we still need a value to instantiate the template.
#define __SYCL_TRAIT_HANDLED_IN_RT 0

namespace sycl {
inline namespace _V1 {

class context;
class device;
class platform;
class kernel_id;
enum class memory_scope;
enum class memory_order;

// TODO: stop using OpenCL directly, use UR.
namespace info {
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  struct Desc {                                                                \
    using return_type = ReturnT;                                               \
  };
// A.1 Platform information desctiptors
namespace platform {
// TODO Despite giving this deprecation warning, we're still yet to implement
// info::device::aspects.
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::get_info() "
                             "with info::device::aspects instead") extensions {
  using return_type = std::vector<std::string>;
  using info_class = sycl::detail::info_class::platform;
  static constexpr ur_platform_info_t ur_code = UR_PLATFORM_INFO_EXTENSIONS;
};
struct profile {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::platform;
  static constexpr ur_platform_info_t ur_code = UR_PLATFORM_INFO_PROFILE;
};
struct version {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::platform;
  static constexpr ur_platform_info_t ur_code = UR_PLATFORM_INFO_VERSION;
};
struct name {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::platform;
  static constexpr ur_platform_info_t ur_code = UR_PLATFORM_INFO_NAME;
};
struct vendor {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::platform;
  static constexpr ur_platform_info_t ur_code = UR_PLATFORM_INFO_VENDOR_NAME;
};
} // namespace platform
// A.2 Context information desctiptors
namespace context {
struct reference_count {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::context;
  static constexpr ur_context_info_t ur_code = UR_CONTEXT_INFO_REFERENCE_COUNT;
};
struct platform {
  using return_type = sycl::platform;
  using info_class = sycl::detail::info_class::context;
};
struct devices {
  using return_type = std::vector<sycl::device>;
  using info_class = sycl::detail::info_class::context;
  static constexpr ur_context_info_t ur_code = UR_CONTEXT_INFO_DEVICES;
};
struct atomic_memory_order_capabilities {
  using return_type = std::vector<sycl::memory_order>;
  using info_class = sycl::detail::info_class::context;
};
struct atomic_memory_scope_capabilities {
  using return_type = std::vector<sycl::memory_scope>;
  using info_class = sycl::detail::info_class::context;
};
struct atomic_fence_order_capabilities {
  using return_type = std::vector<sycl::memory_order>;
  using info_class = sycl::detail::info_class::context;
};
struct atomic_fence_scope_capabilities {
  using return_type = std::vector<sycl::memory_scope>;
  using info_class = sycl::detail::info_class::context;
};
} // namespace context

// A.3 Device information descriptors
enum class device_type : uint32_t {
  cpu = UR_DEVICE_TYPE_CPU,
  gpu = UR_DEVICE_TYPE_GPU,
  accelerator = 0x10000,
  // TODO: evaluate the need for equivalent UR enums for these types
  custom = 0x10001,
  automatic = 0x10002,
  host = 0x10003,
  all = UR_DEVICE_TYPE_ALL
};

enum class partition_property : intptr_t {
  no_partition = 0,
  partition_equally = UR_DEVICE_PARTITION_EQUALLY,
  partition_by_counts = UR_DEVICE_PARTITION_BY_COUNTS,
  partition_by_affinity_domain = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
  ext_intel_partition_by_cslice = UR_DEVICE_PARTITION_BY_CSLICE
};

enum class partition_affinity_domain : intptr_t {
  not_applicable = 0,
  numa = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
  L4_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE,
  L3_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE,
  L2_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE,
  L1_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE,
  next_partitionable = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : uint32_t {
  denorm = UR_DEVICE_FP_CAPABILITY_FLAG_DENORM,
  inf_nan = UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN,
  round_to_nearest = UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST,
  round_to_zero = UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO,
  round_to_inf = UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF,
  fma = UR_DEVICE_FP_CAPABILITY_FLAG_FMA,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, read_write };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

namespace device {

#define __SYCL_DEVICE_INFO(NAME, RETURN_T, UR_CODE)                            \
  struct NAME {                                                                \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
    static constexpr ur_device_info_t ur_code = UR_CODE;                       \
  };
#define __SYCL_DEVICE_INFO_RT(NAME, RETURN_T)                                  \
  struct NAME {                                                                \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
  };
#define __SYCL_DEVICE_INFO_2020_DEP(NAME, RETURN_T, UR_CODE, MSG)              \
  struct __SYCL2020_DEPRECATED(MSG) NAME {                                     \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
    static constexpr ur_device_info_t ur_code = UR_CODE;                       \
  };
#define __SYCL_DEVICE_INFO_2020_DEP_RT(NAME, RETURN_T, MSG)                    \
  struct __SYCL2020_DEPRECATED(MSG) NAME {                                     \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
  };
#define __SYCL_DEVICE_INFO_DEP(NAME, RETURN_T, UR_CODE, MSG)                   \
  struct __SYCL_DEPRECATED(MSG) NAME {                                         \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
    static constexpr ur_device_info_t ur_code = UR_CODE;                       \
  };
#define __SYCL_DEVICE_INFO_DEP_RT(NAME, RETURN_T, MSG)                         \
  struct __SYCL_DEPRECATED(MSG) NAME {                                         \
    using return_type = RETURN_T;                                              \
    using info_class = sycl::detail::info_class::device;                       \
  };

__SYCL_DEVICE_INFO(device_type, info::device_type, UR_DEVICE_INFO_TYPE)
__SYCL_DEVICE_INFO(vendor_id, uint32_t, UR_DEVICE_INFO_VENDOR_ID)
__SYCL_DEVICE_INFO(max_compute_units, uint32_t,
                   UR_DEVICE_INFO_MAX_COMPUTE_UNITS)
__SYCL_DEVICE_INFO(max_work_item_dimensions, uint32_t,
                   UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS)

template <int Dimensions = 3> struct max_work_item_sizes;
template <> struct max_work_item_sizes<1> {
  using return_type = range<1>;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES;
};
template <> struct max_work_item_sizes<2> {
  using return_type = range<2>;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES;
};
template <> struct max_work_item_sizes<3> {
  using return_type = range<3>;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES;
};

__SYCL_DEVICE_INFO(max_work_group_size, size_t,
                   UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE)
__SYCL_DEVICE_INFO(max_num_sub_groups, uint32_t,
                   UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS)
__SYCL_DEVICE_INFO(sub_group_sizes, std::vector<size_t>,
                   UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL)
__SYCL_DEVICE_INFO(preferred_vector_width_char, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR)
__SYCL_DEVICE_INFO(preferred_vector_width_short, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT)
__SYCL_DEVICE_INFO(preferred_vector_width_int, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT)
__SYCL_DEVICE_INFO(preferred_vector_width_long, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG)
__SYCL_DEVICE_INFO(preferred_vector_width_long_long, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG_LONG)
__SYCL_DEVICE_INFO(preferred_vector_width_float, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT)
__SYCL_DEVICE_INFO(preferred_vector_width_double, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE)
__SYCL_DEVICE_INFO(preferred_vector_width_half, uint32_t,
                   UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF)
__SYCL_DEVICE_INFO(native_vector_width_char, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR)
__SYCL_DEVICE_INFO(native_vector_width_short, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT)
__SYCL_DEVICE_INFO(native_vector_width_int, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT)
__SYCL_DEVICE_INFO(native_vector_width_long, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG)
__SYCL_DEVICE_INFO(native_vector_width_long_long, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG_LONG)
__SYCL_DEVICE_INFO(native_vector_width_float, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT)
__SYCL_DEVICE_INFO(native_vector_width_double, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE)
__SYCL_DEVICE_INFO(native_vector_width_half, uint32_t,
                   UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF)
__SYCL_DEVICE_INFO(max_clock_frequency, uint32_t,
                   UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY)
__SYCL_DEVICE_INFO(address_bits, uint32_t, UR_DEVICE_INFO_ADDRESS_BITS)
__SYCL_DEVICE_INFO(max_mem_alloc_size, uint64_t,
                   UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE)
__SYCL_DEVICE_INFO(max_read_image_args, uint32_t,
                   UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS)
__SYCL_DEVICE_INFO(max_write_image_args, uint32_t,
                   UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS)
__SYCL_DEVICE_INFO(image2d_max_width, size_t, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH)
__SYCL_DEVICE_INFO(image2d_max_height, size_t,
                   UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT)
__SYCL_DEVICE_INFO(image3d_max_width, size_t, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH)
__SYCL_DEVICE_INFO(image3d_max_height, size_t,
                   UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT)
__SYCL_DEVICE_INFO(image3d_max_depth, size_t, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH)
__SYCL_DEVICE_INFO(image_max_buffer_size, size_t,
                   UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE)
__SYCL_DEVICE_INFO(max_samplers, uint32_t, UR_DEVICE_INFO_MAX_SAMPLERS)
__SYCL_DEVICE_INFO(max_parameter_size, size_t,
                   UR_DEVICE_INFO_MAX_PARAMETER_SIZE)
__SYCL_DEVICE_INFO(mem_base_addr_align, uint32_t,
                   UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN)
__SYCL_DEVICE_INFO(half_fp_config, std::vector<info::fp_config>,
                   UR_DEVICE_INFO_HALF_FP_CONFIG)
__SYCL_DEVICE_INFO(single_fp_config, std::vector<info::fp_config>,
                   UR_DEVICE_INFO_SINGLE_FP_CONFIG)
__SYCL_DEVICE_INFO(double_fp_config, std::vector<info::fp_config>,
                   UR_DEVICE_INFO_DOUBLE_FP_CONFIG)
__SYCL_DEVICE_INFO(global_mem_cache_type, info::global_mem_cache_type,
                   UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE)
__SYCL_DEVICE_INFO(global_mem_cache_line_size, uint32_t,
                   UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE)
__SYCL_DEVICE_INFO(global_mem_cache_size, uint64_t,
                   UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE)
__SYCL_DEVICE_INFO(global_mem_size, uint64_t, UR_DEVICE_INFO_GLOBAL_MEM_SIZE)
__SYCL_DEVICE_INFO_2020_DEP(max_constant_buffer_size, uint64_t,
                            UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,
                            "deprecated in SYCL 2020")
__SYCL_DEVICE_INFO_2020_DEP(max_constant_args, uint32_t,
                            UR_DEVICE_INFO_MAX_CONSTANT_ARGS,
                            "deprecated in SYCL 2020")
__SYCL_DEVICE_INFO(local_mem_type, info::local_mem_type,
                   UR_DEVICE_INFO_LOCAL_MEM_TYPE)
__SYCL_DEVICE_INFO(local_mem_size, uint64_t, UR_DEVICE_INFO_LOCAL_MEM_SIZE)
__SYCL_DEVICE_INFO(error_correction_support, bool,
                   UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT)
__SYCL_DEVICE_INFO_2020_DEP(host_unified_memory, bool,
                            UR_DEVICE_INFO_HOST_UNIFIED_MEMORY,
                            "deprecated in SYCL 2020, use device::has() with "
                            "one of the aspect::usm_* aspects instead")
__SYCL_DEVICE_INFO(atomic_memory_order_capabilities,
                   std::vector<sycl::memory_order>,
                   UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES)
__SYCL_DEVICE_INFO(atomic_fence_order_capabilities,
                   std::vector<sycl::memory_order>,
                   UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES)
__SYCL_DEVICE_INFO(atomic_memory_scope_capabilities,
                   std::vector<sycl::memory_scope>,
                   UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES)
__SYCL_DEVICE_INFO(atomic_fence_scope_capabilities,
                   std::vector<sycl::memory_scope>,
                   UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES)
__SYCL_DEVICE_INFO(profiling_timer_resolution, size_t,
                   UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION)
__SYCL_DEVICE_INFO_2020_DEP(is_endian_little, bool,
                            UR_DEVICE_INFO_ENDIAN_LITTLE,
                            "deprecated in SYCL 2020, check the byte order of "
                            "the host system instead, the host and the device "
                            "are required to have the same byte order")
__SYCL_DEVICE_INFO(is_available, bool, UR_DEVICE_INFO_AVAILABLE)
__SYCL_DEVICE_INFO_2020_DEP(is_compiler_available, bool,
                            UR_DEVICE_INFO_COMPILER_AVAILABLE,
                            "deprecated in SYCL 2020, use "
                            "device::has(aspect::online_compiler) instead")
__SYCL_DEVICE_INFO_2020_DEP(is_linker_available, bool,
                            UR_DEVICE_INFO_LINKER_AVAILABLE,
                            "deprecated in SYCL 2020, use "
                            "device::has(aspect::online_linker) instead")
__SYCL_DEVICE_INFO(execution_capabilities,
                   std::vector<info::execution_capability>,
                   UR_DEVICE_INFO_EXECUTION_CAPABILITIES)
__SYCL_DEVICE_INFO_2020_DEP(queue_profiling, bool,
                            UR_DEVICE_INFO_QUEUE_PROPERTIES,
                            "deprecated in SYCL 2020, use "
                            "device::has(aspect::queue_profiling) instead")
// TODO: UR_DEVICE_INFO_FORCE_UINT32 looks wrong here:
__SYCL_DEVICE_INFO(built_in_kernel_ids, std::vector<sycl::kernel_id>,
                   UR_DEVICE_INFO_FORCE_UINT32)
__SYCL_DEVICE_INFO_2020_DEP(built_in_kernels, std::vector<std::string>,
                            UR_DEVICE_INFO_BUILT_IN_KERNELS,
                            "deprecated in SYCL 2020, use "
                            "info::device::built_in_kernel_ids instead")
__SYCL_DEVICE_INFO(platform, sycl::platform, UR_DEVICE_INFO_PLATFORM)
__SYCL_DEVICE_INFO(name, std::string, UR_DEVICE_INFO_NAME)
__SYCL_DEVICE_INFO(vendor, std::string, UR_DEVICE_INFO_VENDOR)
__SYCL_DEVICE_INFO(driver_version, std::string,
                   UR_DEVICE_INFO_DRIVER_VERSION)
__SYCL_DEVICE_INFO_2020_DEP(profile, std::string, UR_DEVICE_INFO_PROFILE,
                            "deprecated in SYCL 2020")
__SYCL_DEVICE_INFO(version, std::string, UR_DEVICE_INFO_VERSION)
__SYCL_DEVICE_INFO(backend_version, std::string,
                   UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION)
__SYCL_DEVICE_INFO_2020_DEP(extensions, std::vector<std::string>,
                            UR_DEVICE_INFO_EXTENSIONS,
                            "deprecated in SYCL 2020, use "
                            "info::device::aspects instead")
__SYCL_DEVICE_INFO_2020_DEP(printf_buffer_size, size_t,
                            UR_DEVICE_INFO_PRINTF_BUFFER_SIZE,
                            "deprecated in SYCL 2020")
__SYCL_DEVICE_INFO_2020_DEP(preferred_interop_user_sync, bool,
                            UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,
                            "deprecated in SYCL 2020")
__SYCL_DEVICE_INFO(partition_max_sub_devices, uint32_t,
                   UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES)
__SYCL_DEVICE_INFO(partition_properties,
                   std::vector<info::partition_property>,
                   UR_DEVICE_INFO_SUPPORTED_PARTITIONS)
__SYCL_DEVICE_INFO(partition_affinity_domains,
                   std::vector<info::partition_affinity_domain>,
                   UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN)
__SYCL_DEVICE_INFO(partition_type_property, info::partition_property,
                   UR_DEVICE_INFO_PARTITION_TYPE)
__SYCL_DEVICE_INFO(partition_type_affinity_domain,
                   info::partition_affinity_domain,
                   UR_DEVICE_INFO_PARTITION_TYPE)

// Has custom specialization in device.cpp.
__SYCL_DEVICE_INFO(parent_device, sycl::device, UR_DEVICE_INFO_PARENT_DEVICE)
__SYCL_DEVICE_INFO(aspects, std::vector<sycl::aspect>,
                   UR_DEVICE_INFO_FORCE_UINT32)
__SYCL_DEVICE_INFO_2020_DEP(image_support, bool, UR_DEVICE_INFO_FORCE_UINT32,
                            "deprecated in SYCL 2020, use "
                            "device::has(aspect::ext_intel_legacy_image) to "
                            "query for SYCL 1.2.1 image support")

// Extensions/deprecated
__SYCL_DEVICE_INFO_DEP(atomic64, bool, UR_DEVICE_INFO_ATOMIC_64,
                       "use sycl::aspect::atomic64 instead")
__SYCL_DEVICE_INFO(reference_count, uint32_t,
                   UR_DEVICE_INFO_REFERENCE_COUNT)
// To be dropped (has alternatives/not needed)
__SYCL_DEVICE_INFO(usm_device_allocations, bool,
                   UR_DEVICE_INFO_USM_DEVICE_SUPPORT)
__SYCL_DEVICE_INFO(usm_host_allocations, bool, UR_DEVICE_INFO_USM_HOST_SUPPORT)
__SYCL_DEVICE_INFO(usm_shared_allocations, bool,
                   UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT)
__SYCL_DEVICE_INFO(usm_restricted_shared_allocations, bool,
                   UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT)
__SYCL_DEVICE_INFO(usm_system_allocations, bool,
                   UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT)
__SYCL_DEVICE_INFO_DEP(image_max_array_size, size_t,
                       UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
                       "support for image arrays has been removed in SYCL "
                       "2020")
// To be dropped (no alternatives)
__SYCL_DEVICE_INFO_DEP_RT(opencl_c_version, std::string,
                          "use device::get_info instead")
// Extensions
__SYCL_DEVICE_INFO(sub_group_independent_forward_progress, bool,
                   UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
__SYCL_DEVICE_INFO(ext_oneapi_srgb, bool, UR_DEVICE_INFO_IMAGE_SRGB)

// Deprecated oneapi/intel extension
// TODO: Remove when possible
__SYCL_DEVICE_INFO_DEP(ext_intel_pci_address, std::string,
                       UR_DEVICE_INFO_PCI_ADDRESS,
                       "use ext::intel::info::device::pci_address instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_eu_count, uint32_t,
                       UR_DEVICE_INFO_GPU_EU_COUNT,
                       "use ext::intel::info::device::gpu_eu_count instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_eu_simd_width, uint32_t,
                       UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
                       "use ext::intel::info::device::gpu_eu_simd_width "
                       "instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_slices, uint32_t,
                       UR_DEVICE_INFO_GPU_EU_SLICES,
                       "use ext::intel::info::device::gpu_slices instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_subslices_per_slice, uint32_t,
                       UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
                       "use ext::intel::info::device::gpu_subslices_per_slice"
                       " instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_eu_count_per_subslice, uint32_t,
                       UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
                       "use ext::intel::info::device::"
                       "gpu_eu_count_per_subslice instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_gpu_hw_threads_per_eu, uint32_t,
                       UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
                       "use ext::intel::info::device::gpu_hw_threads_per_eu "
                       "instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_device_info_uuid, detail::uuid_type,
                       UR_DEVICE_INFO_UUID,
                       "use ext::intel::info::device::uuid instead")
__SYCL_DEVICE_INFO_DEP(ext_intel_max_mem_bandwidth, uint64_t,
                       UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH,
                       "use ext::intel::info::device::max_mem_bandwidth "
                       "instead")

__SYCL_DEVICE_INFO_DEP_RT(ext_oneapi_max_work_groups_1d, id<1>,
                          "use ext::oneapi::experimental::info::device::"
                          "max_work_groups<1> instead")
__SYCL_DEVICE_INFO_DEP_RT(ext_oneapi_max_work_groups_2d, id<2>,
                          "use ext::oneapi::experimental::info::device::"
                          "max_work_groups<2> instead")
__SYCL_DEVICE_INFO_DEP(ext_oneapi_max_work_groups_3d, id<3>,
                       UR_DEVICE_INFO_MAX_WORK_GROUPS_3D,
                       "use ext::oneapi::experimental::info::device::"
                       "max_work_groups<3> instead")
__SYCL_DEVICE_INFO_RT(ext_oneapi_max_global_work_groups, size_t)
__SYCL_DEVICE_INFO_RT(ext_oneapi_cuda_cluster_group, bool)

#undef __SYCL_DEVICE_INFO
#undef __SYCL_DEVICE_INFO_RT
#undef __SYCL_DEVICE_INFO_2020_DEP
#undef __SYCL_DEVICE_INFO_2020_DEP_RT
#undef __SYCL_DEVICE_INFO_DEP
#undef __SYCL_DEVICE_INFO_DEP_RT
} // namespace device

// A.4 Queue information descriptors
namespace queue {
struct context {
  using return_type = sycl::context;
  using info_class = sycl::detail::info_class::queue;
  static constexpr ur_queue_info_t ur_code = UR_QUEUE_INFO_CONTEXT;
};
struct device {
  using return_type = sycl::device;
  using info_class = sycl::detail::info_class::queue;
  static constexpr ur_queue_info_t ur_code = UR_QUEUE_INFO_DEVICE;
};
struct reference_count {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::queue;
  static constexpr ur_queue_info_t ur_code = UR_QUEUE_INFO_REFERENCE_COUNT;
};
} // namespace queue

// A.5 Kernel information desctiptors
namespace kernel {
struct num_args {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_NUM_ARGS;
};
struct attributes {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::kernel;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_ATTRIBUTES;
};
struct function_name {
  using return_type = std::string;
  using info_class = sycl::detail::info_class::kernel;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_FUNCTION_NAME;
};
struct reference_count {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_REFERENCE_COUNT;
};
struct context {
  using return_type = sycl::context;
  using info_class = sycl::detail::info_class::kernel;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_CONTEXT;
};
} // namespace kernel

namespace kernel_device_specific {
struct global_work_size {
  using return_type = sycl::range<3>;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_group_info_t ur_code =
      UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE;
};
struct work_group_size {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_group_info_t ur_code =
      UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE;
};
struct compile_work_group_size {
  using return_type = sycl::range<3>;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_group_info_t ur_code =
      UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE;
};
struct preferred_work_group_size_multiple {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_group_info_t ur_code =
      UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
};
struct private_mem_size {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_group_info_t ur_code =
      UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE;
};
// The next five traits dispatch through different UR APIs than the four
// above (urKernelGetSubGroupInfo / urKernelGetInfo rather than
// urKernelGetGroupInfo). They each carry their own native UR enum type;
// info_class::kernel_device_specific has ur_code_type = void to permit the
// mismatch.
struct max_num_sub_groups {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_sub_group_info_t ur_code =
      UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS;
};
struct compile_num_sub_groups {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_sub_group_info_t ur_code =
      UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS;
};
struct max_sub_group_size {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_sub_group_info_t ur_code =
      UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE;
};
struct compile_sub_group_size {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_sub_group_info_t ur_code =
      UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL;
};
struct ext_codeplay_num_regs {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_NUM_REGS;
};
} // namespace kernel_device_specific

// A.6 Event information desctiptors
enum class event_command_status : int32_t {
  submitted = UR_EVENT_STATUS_SUBMITTED,
  running = UR_EVENT_STATUS_RUNNING,
  complete = UR_EVENT_STATUS_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};

namespace event {
struct command_execution_status {
  using return_type = info::event_command_status;
  using info_class = sycl::detail::info_class::event;
  static constexpr ur_event_info_t ur_code =
      UR_EVENT_INFO_COMMAND_EXECUTION_STATUS;
};
struct reference_count {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::event;
  static constexpr ur_event_info_t ur_code = UR_EVENT_INFO_REFERENCE_COUNT;
};
} // namespace event
namespace event_profiling {
struct command_submit {
  using return_type = uint64_t;
  using info_class = sycl::detail::info_class::event_profiling;
  static constexpr ur_profiling_info_t ur_code =
      UR_PROFILING_INFO_COMMAND_SUBMIT;
};
struct command_start {
  using return_type = uint64_t;
  using info_class = sycl::detail::info_class::event_profiling;
  static constexpr ur_profiling_info_t ur_code =
      UR_PROFILING_INFO_COMMAND_START;
};
struct command_end {
  using return_type = uint64_t;
  using info_class = sycl::detail::info_class::event_profiling;
  static constexpr ur_profiling_info_t ur_code = UR_PROFILING_INFO_COMMAND_END;
};
} // namespace event_profiling
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace info

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  namespace Namespace {                                                        \
  namespace info {                                                             \
  namespace DescType {                                                         \
  struct Desc {                                                                \
    using return_type = ReturnT;                                               \
  };                                                                           \
  } /*DescType*/                                                               \
  } /*info*/                                                                   \
  } /*Namespace*/

#define __SYCL_PARAM_TRAITS_TEMPLATE_SPEC(Namespace, DescType, Desc, ReturnT,  \
                                          UrCode)                              \
  namespace Namespace {                                                        \
  namespace info {                                                             \
  namespace DescType {                                                         \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };                                                                           \
  } /*namespace DescType */                                                    \
  } /*namespace info */                                                        \
  } /*namespace Namespace */

#define __SYCL_PARAM_TRAITS_TEMPLATE_PARTIAL_SPEC(Namespace, Desctype, Desc,   \
                                                  ReturnT, UrCode)             \
  namespace Namespace::info {                                                  \
  namespace Desctype {                                                         \
  template <int Dimensions> struct Desc {                                      \
    using return_type = ReturnT<Dimensions>;                                   \
  };                                                                           \
  }                                                                            \
  }

#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_intel_kernel_info_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#include <sycl/info/ext_oneapi_kernel_queue_specific_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_PARTIAL_SPEC
} // namespace _V1
} // namespace sycl
