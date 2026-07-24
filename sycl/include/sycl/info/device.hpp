//==----- device.hpp - SYCL device information descriptors ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/device_info_types.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/range.hpp>
#include <unified-runtime/ur_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;
class platform;
class kernel_id;
enum class memory_scope;
enum class memory_order;

namespace info {

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

template <ur_device_info_t UrCode>
using device_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::device, UrCode>;
using device_runtime_traits =
    sycl::detail::rt_traits_base<sycl::detail::info_class::device>;

struct device_type : device_traits<UR_DEVICE_INFO_TYPE> {
  using return_type = info::device_type;
};
struct vendor_id : device_traits<UR_DEVICE_INFO_VENDOR_ID> {
  using return_type = uint32_t;
};
struct max_compute_units : device_traits<UR_DEVICE_INFO_MAX_COMPUTE_UNITS> {
  using return_type = uint32_t;
};
struct max_work_item_dimensions
    : device_traits<UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS> {
  using return_type = uint32_t;
};

template <int Dimensions = 3> struct max_work_item_sizes;
template <>
struct max_work_item_sizes<1>
    : device_traits<UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES> {
  using return_type = range<1>;
};
template <>
struct max_work_item_sizes<2>
    : device_traits<UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES> {
  using return_type = range<2>;
};
template <>
struct max_work_item_sizes<3>
    : device_traits<UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES> {
  using return_type = range<3>;
};

struct max_work_group_size : device_traits<UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE> {
  using return_type = size_t;
};
struct max_num_sub_groups : device_traits<UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS> {
  using return_type = uint32_t;
};
struct sub_group_sizes : device_traits<UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL> {
  using return_type = std::vector<size_t>;
};
struct preferred_vector_width_char
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR> {
  using return_type = uint32_t;
};
struct preferred_vector_width_short
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT> {
  using return_type = uint32_t;
};
struct preferred_vector_width_int
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT> {
  using return_type = uint32_t;
};
struct preferred_vector_width_long
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG> {
  using return_type = uint32_t;
};
struct preferred_vector_width_long_long
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG_LONG> {
  using return_type = uint32_t;
};
struct preferred_vector_width_float
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT> {
  using return_type = uint32_t;
};
struct preferred_vector_width_double
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE> {
  using return_type = uint32_t;
};
struct preferred_vector_width_half
    : device_traits<UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF> {
  using return_type = uint32_t;
};
struct native_vector_width_char
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR> {
  using return_type = uint32_t;
};
struct native_vector_width_short
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT> {
  using return_type = uint32_t;
};
struct native_vector_width_int
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT> {
  using return_type = uint32_t;
};
struct native_vector_width_long
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG> {
  using return_type = uint32_t;
};
struct native_vector_width_long_long
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG_LONG> {
  using return_type = uint32_t;
};
struct native_vector_width_float
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT> {
  using return_type = uint32_t;
};
struct native_vector_width_double
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE> {
  using return_type = uint32_t;
};
struct native_vector_width_half
    : device_traits<UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF> {
  using return_type = uint32_t;
};
struct max_clock_frequency : device_traits<UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY> {
  using return_type = uint32_t;
};
struct address_bits : device_traits<UR_DEVICE_INFO_ADDRESS_BITS> {
  using return_type = uint32_t;
};
struct max_mem_alloc_size : device_traits<UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE> {
  using return_type = uint64_t;
};
struct max_read_image_args : device_traits<UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS> {
  using return_type = uint32_t;
};
struct max_write_image_args
    : device_traits<UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS> {
  using return_type = uint32_t;
};
struct image2d_max_width : device_traits<UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH> {
  using return_type = size_t;
};
struct image2d_max_height : device_traits<UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT> {
  using return_type = size_t;
};
struct image3d_max_width : device_traits<UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH> {
  using return_type = size_t;
};
struct image3d_max_height : device_traits<UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT> {
  using return_type = size_t;
};
struct image3d_max_depth : device_traits<UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH> {
  using return_type = size_t;
};
struct image_max_buffer_size
    : device_traits<UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE> {
  using return_type = size_t;
};
struct max_samplers : device_traits<UR_DEVICE_INFO_MAX_SAMPLERS> {
  using return_type = uint32_t;
};
struct max_parameter_size : device_traits<UR_DEVICE_INFO_MAX_PARAMETER_SIZE> {
  using return_type = size_t;
};
struct mem_base_addr_align : device_traits<UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN> {
  using return_type = uint32_t;
};
struct half_fp_config : device_traits<UR_DEVICE_INFO_HALF_FP_CONFIG> {
  using return_type = std::vector<info::fp_config>;
};
struct single_fp_config : device_traits<UR_DEVICE_INFO_SINGLE_FP_CONFIG> {
  using return_type = std::vector<info::fp_config>;
};
struct double_fp_config : device_traits<UR_DEVICE_INFO_DOUBLE_FP_CONFIG> {
  using return_type = std::vector<info::fp_config>;
};
struct global_mem_cache_type
    : device_traits<UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE> {
  using return_type = info::global_mem_cache_type;
};
struct global_mem_cache_line_size
    : device_traits<UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE> {
  using return_type = uint32_t;
};
struct global_mem_cache_size
    : device_traits<UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE> {
  using return_type = uint64_t;
};
struct global_mem_size : device_traits<UR_DEVICE_INFO_GLOBAL_MEM_SIZE> {
  using return_type = uint64_t;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") max_constant_buffer_size
    : device_traits<UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE> {
  using return_type = uint64_t;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") max_constant_args
    : device_traits<UR_DEVICE_INFO_MAX_CONSTANT_ARGS> {
  using return_type = uint32_t;
};
struct local_mem_type : device_traits<UR_DEVICE_INFO_LOCAL_MEM_TYPE> {
  using return_type = info::local_mem_type;
};
struct local_mem_size : device_traits<UR_DEVICE_INFO_LOCAL_MEM_SIZE> {
  using return_type = uint64_t;
};
struct error_correction_support
    : device_traits<UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT> {
  using return_type = bool;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::has() with "
                             "one of the aspect::usm_* aspects instead")
    host_unified_memory : device_traits<UR_DEVICE_INFO_HOST_UNIFIED_MEMORY> {
  using return_type = bool;
};
struct atomic_memory_order_capabilities
    : device_traits<UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES> {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_fence_order_capabilities
    : device_traits<UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES> {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_memory_scope_capabilities
    : device_traits<UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES> {
  using return_type = std::vector<sycl::memory_scope>;
};
struct atomic_fence_scope_capabilities
    : device_traits<UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES> {
  using return_type = std::vector<sycl::memory_scope>;
};
struct profiling_timer_resolution
    : device_traits<UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION> {
  using return_type = size_t;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, check the byte order of "
                             "the host system instead, the host and the device "
                             "are required to have the same byte order")
    is_endian_little : device_traits<UR_DEVICE_INFO_ENDIAN_LITTLE> {
  using return_type = bool;
};
struct is_available : device_traits<UR_DEVICE_INFO_AVAILABLE> {
  using return_type = bool;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "device::has(aspect::online_compiler) instead")
    is_compiler_available : device_traits<UR_DEVICE_INFO_COMPILER_AVAILABLE> {
  using return_type = bool;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "device::has(aspect::online_linker) instead")
    is_linker_available : device_traits<UR_DEVICE_INFO_LINKER_AVAILABLE> {
  using return_type = bool;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") execution_capabilities
    : device_traits<UR_DEVICE_INFO_EXECUTION_CAPABILITIES> {
  using return_type = std::vector<info::execution_capability>;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "device::has(aspect::queue_profiling) instead")
    queue_profiling : device_traits<UR_DEVICE_INFO_QUEUE_PROPERTIES> {
  using return_type = bool;
};
// TODO: UR_DEVICE_INFO_FORCE_UINT32 looks wrong here:
struct built_in_kernel_ids : device_traits<UR_DEVICE_INFO_FORCE_UINT32> {
  using return_type = std::vector<sycl::kernel_id>;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "info::device::built_in_kernel_ids instead")
    built_in_kernels : device_traits<UR_DEVICE_INFO_BUILT_IN_KERNELS> {
  using return_type = std::vector<std::string>;
};
struct platform : device_traits<UR_DEVICE_INFO_PLATFORM> {
  using return_type = sycl::platform;
};
struct name : device_traits<UR_DEVICE_INFO_NAME> {
  using return_type = std::string;
};
struct vendor : device_traits<UR_DEVICE_INFO_VENDOR> {
  using return_type = std::string;
};
struct driver_version : device_traits<UR_DEVICE_INFO_DRIVER_VERSION> {
  using return_type = std::string;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") profile
    : device_traits<UR_DEVICE_INFO_PROFILE> {
  using return_type = std::string;
};
struct version : device_traits<UR_DEVICE_INFO_VERSION> {
  using return_type = std::string;
};
struct backend_version : device_traits<UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION> {
  using return_type = std::string;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "info::device::aspects instead") extensions
    : device_traits<UR_DEVICE_INFO_EXTENSIONS> {
  using return_type = std::vector<std::string>;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") printf_buffer_size
    : device_traits<UR_DEVICE_INFO_PRINTF_BUFFER_SIZE> {
  using return_type = size_t;
};
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
    preferred_interop_user_sync
    : device_traits<UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC> {
  using return_type = bool;
};
struct partition_max_sub_devices
    : device_traits<UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES> {
  using return_type = uint32_t;
};
struct partition_properties
    : device_traits<UR_DEVICE_INFO_SUPPORTED_PARTITIONS> {
  using return_type = std::vector<info::partition_property>;
};
struct partition_affinity_domains
    : device_traits<UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN> {
  using return_type = std::vector<info::partition_affinity_domain>;
};
struct partition_type_property : device_traits<UR_DEVICE_INFO_PARTITION_TYPE> {
  using return_type = info::partition_property;
};
struct partition_type_affinity_domain
    : device_traits<UR_DEVICE_INFO_PARTITION_TYPE> {
  using return_type = info::partition_affinity_domain;
};

// Has custom specialization in device.cpp.
struct parent_device : device_traits<UR_DEVICE_INFO_PARENT_DEVICE> {
  using return_type = sycl::device;
};
struct aspects : device_traits<UR_DEVICE_INFO_FORCE_UINT32> {
  using return_type = std::vector<sycl::aspect>;
};

struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use "
                             "device::has(aspect::ext_intel_legacy_image) to "
                             "query for SYCL 1.2.1 image support") image_support
    : device_traits<UR_DEVICE_INFO_FORCE_UINT32> {
  using return_type = bool;
};

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
struct __SYCL_DEPRECATED("use sycl::aspect::atomic64 instead") atomic64
    : device_traits<UR_DEVICE_INFO_ATOMIC_64> {
  using return_type = bool;
};
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
struct __SYCL_DEPRECATED("info::device::reference_count is not part of "
                         "SYCL 2020") reference_count
    : device_traits<UR_DEVICE_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
// To be dropped (has alternatives/not needed)
struct __SYCL_DEPRECATED("use sycl::aspect::usm_device_allocations instead")
    usm_device_allocations : device_traits<UR_DEVICE_INFO_USM_DEVICE_SUPPORT> {
  using return_type = bool;
};
struct __SYCL_DEPRECATED("use sycl::aspect::usm_host_allocations instead")
    usm_host_allocations : device_traits<UR_DEVICE_INFO_USM_HOST_SUPPORT> {
  using return_type = bool;
};
struct __SYCL_DEPRECATED("use sycl::aspect::usm_shared_allocations instead")
    usm_shared_allocations
    : device_traits<UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT> {
  using return_type = bool;
};
struct __SYCL_DEPRECATED("deprecated descriptor")
    usm_restricted_shared_allocations
    : device_traits<UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT> {
  using return_type = bool;
};
struct __SYCL_DEPRECATED("use sycl::aspect::usm_system_allocations instead")
    usm_system_allocations
    : device_traits<UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT> {
  using return_type = bool;
};
struct __SYCL_DEPRECATED("support for image arrays has been removed in SYCL "
                         "2020") image_max_array_size
    : device_traits<UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE> {
  using return_type = size_t;
};
// To be dropped (no alternatives)
struct __SYCL_DEPRECATED("use device::get_info instead") opencl_c_version
    : device_runtime_traits {
  using return_type = std::string;
};
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
// Extensions
struct __SYCL_DEPRECATED("extension is deprecated")
    sub_group_independent_forward_progress
    : device_traits<UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS> {
  using return_type = bool;
};
struct ext_oneapi_srgb : device_traits<UR_DEVICE_INFO_IMAGE_SRGB> {
  using return_type = bool;
};

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
// Deprecated oneapi/intel extension
struct __SYCL_DEPRECATED("use ext::intel::info::device::pci_address instead")
    ext_intel_pci_address : device_traits<UR_DEVICE_INFO_PCI_ADDRESS> {
  using return_type = std::string;
};
struct __SYCL_DEPRECATED("use ext::intel::info::device::gpu_eu_count instead")
    ext_intel_gpu_eu_count : device_traits<UR_DEVICE_INFO_GPU_EU_COUNT> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED(
    "use ext::intel::info::device::gpu_eu_simd_width instead")
    ext_intel_gpu_eu_simd_width
    : device_traits<UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED("use ext::intel::info::device::gpu_slices instead")
    ext_intel_gpu_slices : device_traits<UR_DEVICE_INFO_GPU_EU_SLICES> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED(
    "use ext::intel::info::device::gpu_subslices_per_slice instead")
    ext_intel_gpu_subslices_per_slice
    : device_traits<UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED(
    "use ext::intel::info::device::gpu_eu_count_per_subslice instead")
    ext_intel_gpu_eu_count_per_subslice
    : device_traits<UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED(
    "use ext::intel::info::device::gpu_hw_threads_per_eu instead")
    ext_intel_gpu_hw_threads_per_eu
    : device_traits<UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU> {
  using return_type = uint32_t;
};
struct __SYCL_DEPRECATED("use ext::intel::info::device::uuid instead")
    ext_intel_device_info_uuid : device_traits<UR_DEVICE_INFO_UUID> {
  using return_type = detail::uuid_type;
};
struct __SYCL_DEPRECATED(
    "use ext::intel::info::device::max_mem_bandwidth instead")
    ext_intel_max_mem_bandwidth
    : device_traits<UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH> {
  using return_type = uint64_t;
};

struct __SYCL_DEPRECATED(
    "use ext::oneapi::experimental::info::device::max_work_groups<1> instead")
    ext_oneapi_max_work_groups_1d : device_runtime_traits {
  using return_type = id<1>;
};
struct __SYCL_DEPRECATED(
    "use ext::oneapi::experimental::info::device::max_work_groups<2> instead")
    ext_oneapi_max_work_groups_2d : device_runtime_traits {
  using return_type = id<2>;
};
struct __SYCL_DEPRECATED(
    "use ext::oneapi::experimental::info::device::max_work_groups<3> instead")
    ext_oneapi_max_work_groups_3d
    : device_traits<UR_DEVICE_INFO_MAX_WORK_GROUPS_3D> {
  using return_type = id<3>;
};

struct __SYCL_DEPRECATED(
    "use sycl::ext::oneapi::experimental::info::max_global_work_groups "
    "instead") ext_oneapi_max_global_work_groups : device_runtime_traits {
  using return_type = size_t;
};

#endif // __INTEL_PREVIEW_BREAKING_CHANGES
struct ext_oneapi_cuda_cluster_group : device_runtime_traits {
  using return_type = bool;
};

} // namespace device

} // namespace info

namespace detail {
// SFINAE predicate confining `device::get_info<T>()` to device traits.
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_device_info_desc : is_info_desc_for<T, info_class::device> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
