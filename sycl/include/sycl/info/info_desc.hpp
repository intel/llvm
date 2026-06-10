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
// A.1 Platform information desctiptors
namespace platform {
template <ur_platform_info_t UrCode>
using platform_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::platform, UrCode>;

// TODO Despite giving this deprecation warning, we're still yet to implement
// info::device::aspects.
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::get_info() "
                             "with info::device::aspects instead") extensions
    : platform_traits<UR_PLATFORM_INFO_EXTENSIONS> {
  using return_type = std::vector<std::string>;
};
struct profile : platform_traits<UR_PLATFORM_INFO_PROFILE> {
  using return_type = std::string;
};
struct version : platform_traits<UR_PLATFORM_INFO_VERSION> {
  using return_type = std::string;
};
struct name : platform_traits<UR_PLATFORM_INFO_NAME> {
  using return_type = std::string;
};
struct vendor : platform_traits<UR_PLATFORM_INFO_VENDOR_NAME> {
  using return_type = std::string;
};
} // namespace platform
// A.2 Context information desctiptors
namespace context {
template <ur_context_info_t UrCode>
using context_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::context, UrCode>;
using context_runtime_traits =
    sycl::detail::rt_traits_base<sycl::detail::info_class::context>;

struct reference_count : context_traits<UR_CONTEXT_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
struct platform : context_runtime_traits {
  using return_type = sycl::platform;
};
struct devices : context_traits<UR_CONTEXT_INFO_DEVICES> {
  using return_type = std::vector<sycl::device>;
};
struct atomic_memory_order_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_memory_scope_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_scope>;
};
struct atomic_fence_order_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_fence_scope_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_scope>;
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
struct execution_capabilities
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

// Extensions/deprecated
struct __SYCL_DEPRECATED("use sycl::aspect::atomic64 instead") atomic64
    : device_traits<UR_DEVICE_INFO_ATOMIC_64> {
  using return_type = bool;
};
struct reference_count : device_traits<UR_DEVICE_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
// To be dropped (has alternatives/not needed)
struct usm_device_allocations
    : device_traits<UR_DEVICE_INFO_USM_DEVICE_SUPPORT> {
  using return_type = bool;
};
struct usm_host_allocations : device_traits<UR_DEVICE_INFO_USM_HOST_SUPPORT> {
  using return_type = bool;
};
struct usm_shared_allocations
    : device_traits<UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT> {
  using return_type = bool;
};
struct usm_restricted_shared_allocations
    : device_traits<UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT> {
  using return_type = bool;
};
struct usm_system_allocations
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
// Extensions
struct sub_group_independent_forward_progress
    : device_traits<UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS> {
  using return_type = bool;
};
struct ext_oneapi_srgb : device_traits<UR_DEVICE_INFO_IMAGE_SRGB> {
  using return_type = bool;
};

// Deprecated oneapi/intel extension
// TODO: Remove when possible
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
struct ext_oneapi_max_global_work_groups : device_runtime_traits {
  using return_type = size_t;
};
struct ext_oneapi_cuda_cluster_group : device_runtime_traits {
  using return_type = bool;
};

} // namespace device

// A.4 Queue information descriptors
namespace queue {
template <ur_queue_info_t UrCode>
using queue_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::queue, UrCode>;

struct context : queue_traits<UR_QUEUE_INFO_CONTEXT> {
  using return_type = sycl::context;
};
struct device : queue_traits<UR_QUEUE_INFO_DEVICE> {
  using return_type = sycl::device;
};
struct reference_count : queue_traits<UR_QUEUE_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
} // namespace queue

// A.5 Kernel information desctiptors
namespace kernel {
template <ur_kernel_info_t UrCode>
using kernel_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::kernel, UrCode>;

struct num_args : kernel_traits<UR_KERNEL_INFO_NUM_ARGS> {
  using return_type = uint32_t;
};
struct attributes : kernel_traits<UR_KERNEL_INFO_ATTRIBUTES> {
  using return_type = std::string;
};
struct function_name : kernel_traits<UR_KERNEL_INFO_FUNCTION_NAME> {
  using return_type = std::string;
};
struct reference_count : kernel_traits<UR_KERNEL_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
struct context : kernel_traits<UR_KERNEL_INFO_CONTEXT> {
  using return_type = sycl::context;
};
} // namespace kernel

namespace kernel_device_specific {
// kernel_device_specific traits dispatch through three UR APIs and so mix
// three native UR enum families; info_class::kernel_device_specific has
// ur_code_type = void to permit the mismatch. Use the matching alias for
// each family.
template <ur_kernel_group_info_t UrCode>
using group_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;
template <ur_kernel_sub_group_info_t UrCode>
using sub_group_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;
template <ur_kernel_info_t UrCode>
using kernel_info_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;

struct global_work_size : group_traits<UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE> {
  using return_type = sycl::range<3>;
};
struct work_group_size : group_traits<UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE> {
  using return_type = size_t;
};
struct compile_work_group_size
    : group_traits<UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE> {
  using return_type = sycl::range<3>;
};
struct preferred_work_group_size_multiple
    : group_traits<UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> {
  using return_type = size_t;
};
struct private_mem_size : group_traits<UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE> {
  using return_type = size_t;
};
struct max_num_sub_groups
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS> {
  using return_type = uint32_t;
};
struct compile_num_sub_groups
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS> {
  using return_type = uint32_t;
};
struct max_sub_group_size
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE> {
  using return_type = uint32_t;
};
struct compile_sub_group_size
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL> {
  using return_type = uint32_t;
};
struct ext_codeplay_num_regs : kernel_info_traits<UR_KERNEL_INFO_NUM_REGS> {
  using return_type = uint32_t;
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
template <ur_event_info_t UrCode>
using event_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::event, UrCode>;

struct command_execution_status
    : event_traits<UR_EVENT_INFO_COMMAND_EXECUTION_STATUS> {
  using return_type = info::event_command_status;
};
struct reference_count : event_traits<UR_EVENT_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
} // namespace event
namespace event_profiling {
template <ur_profiling_info_t UrCode>
using profiling_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::event_profiling,
                                 UrCode>;

struct command_submit : profiling_traits<UR_PROFILING_INFO_COMMAND_SUBMIT> {
  using return_type = uint64_t;
};
struct command_start : profiling_traits<UR_PROFILING_INFO_COMMAND_START> {
  using return_type = uint64_t;
};
struct command_end : profiling_traits<UR_PROFILING_INFO_COMMAND_END> {
  using return_type = uint64_t;
};
} // namespace event_profiling

} // namespace info

} // namespace _V1
} // namespace sycl
