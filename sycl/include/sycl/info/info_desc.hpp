//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/id.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

#ifdef __SYCL_INTERNAL_API
class program;
#endif
class device;
class platform;
class kernel_id;
enum class memory_scope;

// TODO: stop using OpenCL directly, use PI.
namespace info {

// Information descriptors
// A.1 Platform information descriptors
enum class platform {
  profile = PI_PLATFORM_INFO_PROFILE,
  version = PI_PLATFORM_INFO_VERSION,
  name = PI_PLATFORM_INFO_NAME,
  vendor = PI_PLATFORM_INFO_VENDOR,
  extensions __SYCL2020_DEPRECATED(
      "platform::extensions is deprecated, use device::get_info() with"
      " info::device::aspects instead.") = PI_PLATFORM_INFO_EXTENSIONS,
};

// A.2 Context information desctiptors
enum class context : pi_uint32 {
  reference_count = PI_CONTEXT_INFO_REFERENCE_COUNT,
  platform = PI_CONTEXT_INFO_PLATFORM,
  devices = PI_CONTEXT_INFO_DEVICES,
  atomic_memory_order_capabilities =
      PI_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
  atomic_memory_scope_capabilities =
      PI_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
};

// A.3 Device information descriptors
enum class device : pi_uint32 {
  device_type = PI_DEVICE_INFO_TYPE,
  vendor_id = PI_DEVICE_INFO_VENDOR_ID,
  max_compute_units = PI_DEVICE_INFO_MAX_COMPUTE_UNITS,
  max_work_item_dimensions = PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
  max_work_item_sizes = PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
  max_work_group_size = PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,

  preferred_vector_width_char = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR,
  preferred_vector_width_short = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT,
  preferred_vector_width_int = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT,
  preferred_vector_width_long = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG,
  preferred_vector_width_float = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT,
  preferred_vector_width_double = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE,
  preferred_vector_width_half = PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF,

  native_vector_width_char = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
  native_vector_width_short = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
  native_vector_width_int = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
  native_vector_width_long = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
  native_vector_width_float = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
  native_vector_width_double = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
  native_vector_width_half = PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,

  max_clock_frequency = PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY,
  address_bits = PI_DEVICE_INFO_ADDRESS_BITS,
  max_mem_alloc_size = PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
  image_support = PI_DEVICE_INFO_IMAGE_SUPPORT,
  max_read_image_args = PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS,
  max_write_image_args = PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,
  image2d_max_width = PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH,
  image2d_max_height = PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT,
  image3d_max_width = PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH,
  image3d_max_height = PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT,
  image3d_max_depth = PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH,
  image_max_buffer_size = PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
  image_max_array_size = PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
  max_samplers = PI_DEVICE_INFO_MAX_SAMPLERS,
  max_parameter_size = PI_DEVICE_INFO_MAX_PARAMETER_SIZE,
  mem_base_addr_align = PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,
  half_fp_config = PI_DEVICE_INFO_HALF_FP_CONFIG,
  single_fp_config = PI_DEVICE_INFO_SINGLE_FP_CONFIG,
  double_fp_config = PI_DEVICE_INFO_DOUBLE_FP_CONFIG,
  global_mem_cache_type = PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE,
  global_mem_cache_line_size = PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE,
  global_mem_cache_size = PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,
  global_mem_size = PI_DEVICE_INFO_GLOBAL_MEM_SIZE,
  max_constant_buffer_size __SYCL2020_DEPRECATED(
      "max_constant_buffer_size is deprecated") =
      PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,
  max_constant_args __SYCL2020_DEPRECATED("max_constant_args is deprecated") =
      PI_DEVICE_INFO_MAX_CONSTANT_ARGS,
  local_mem_type = PI_DEVICE_INFO_LOCAL_MEM_TYPE,
  local_mem_size = PI_DEVICE_INFO_LOCAL_MEM_SIZE,
  error_correction_support = PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,
  host_unified_memory = PI_DEVICE_INFO_HOST_UNIFIED_MEMORY,
  profiling_timer_resolution = PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION,
  is_endian_little = PI_DEVICE_INFO_ENDIAN_LITTLE,
  is_available = PI_DEVICE_INFO_AVAILABLE,
  is_compiler_available = PI_DEVICE_INFO_COMPILER_AVAILABLE,
  is_linker_available = PI_DEVICE_INFO_LINKER_AVAILABLE,
  execution_capabilities = PI_DEVICE_INFO_EXECUTION_CAPABILITIES,
  queue_profiling = PI_DEVICE_INFO_QUEUE_PROPERTIES,
  built_in_kernels __SYCL2020_DEPRECATED("use built_in_kernel_ids instead") =
      PI_DEVICE_INFO_BUILT_IN_KERNELS,
  platform = PI_DEVICE_INFO_PLATFORM,
  name = PI_DEVICE_INFO_NAME,
  vendor = PI_DEVICE_INFO_VENDOR,
  driver_version = PI_DEVICE_INFO_DRIVER_VERSION,
  profile = PI_DEVICE_INFO_PROFILE,
  version = PI_DEVICE_INFO_VERSION,
  opencl_c_version = PI_DEVICE_INFO_OPENCL_C_VERSION,
  extensions __SYCL2020_DEPRECATED(
      "device::extensions is deprecated, use info::device::aspects"
      " instead.") = PI_DEVICE_INFO_EXTENSIONS,
  printf_buffer_size = PI_DEVICE_INFO_PRINTF_BUFFER_SIZE,
  preferred_interop_user_sync = PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,
  parent_device = PI_DEVICE_INFO_PARENT_DEVICE,
  partition_max_sub_devices = PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES,
  partition_properties = PI_DEVICE_INFO_PARTITION_PROPERTIES,
  partition_affinity_domains = PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,
  partition_type_affinity_domain = PI_DEVICE_INFO_PARTITION_TYPE,
  reference_count = PI_DEVICE_INFO_REFERENCE_COUNT,
  il_version = PI_DEVICE_INFO_IL_VERSION,
  max_num_sub_groups = PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS,
  sub_group_independent_forward_progress =
      PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
  sub_group_sizes = PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
  partition_type_property,
  kernel_kernel_pipe_support,
  built_in_kernel_ids,
  backend_version = PI_DEVICE_INFO_BACKEND_VERSION,
  // USM
  usm_device_allocations = PI_USM_DEVICE_SUPPORT,
  usm_host_allocations = PI_USM_HOST_SUPPORT,
  usm_shared_allocations = PI_USM_SINGLE_SHARED_SUPPORT,
  usm_restricted_shared_allocations = PI_USM_CROSS_SHARED_SUPPORT,
  usm_system_allocations = PI_USM_SYSTEM_SHARED_SUPPORT,
  usm_system_allocator __SYCL2020_DEPRECATED(
      "use usm_system_allocations instead") = usm_system_allocations,

  // intel extensions
  ext_intel_pci_address = PI_DEVICE_INFO_PCI_ADDRESS,
  ext_intel_gpu_eu_count = PI_DEVICE_INFO_GPU_EU_COUNT,
  ext_intel_gpu_eu_simd_width = PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
  ext_intel_gpu_slices = PI_DEVICE_INFO_GPU_SLICES,
  ext_intel_gpu_subslices_per_slice = PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
  ext_intel_gpu_eu_count_per_subslice =
      PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
  ext_intel_gpu_hw_threads_per_eu = PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
  ext_intel_max_mem_bandwidth = PI_DEVICE_INFO_MAX_MEM_BANDWIDTH,
  ext_intel_mem_channel = PI_MEM_PROPERTIES_CHANNEL,
  ext_oneapi_srgb = PI_DEVICE_INFO_IMAGE_SRGB,
  ext_intel_device_info_uuid = PI_DEVICE_INFO_UUID,
  atomic64 = PI_DEVICE_INFO_ATOMIC_64,
  atomic_memory_order_capabilities =
      PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
  ext_oneapi_max_global_work_groups =
      PI_EXT_ONEAPI_DEVICE_INFO_MAX_GLOBAL_WORK_GROUPS,
  ext_oneapi_max_work_groups_1d = PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_1D,
  ext_oneapi_max_work_groups_2d = PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_2D,
  ext_oneapi_max_work_groups_3d = PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D,
  atomic_memory_scope_capabilities =
      PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
  ext_oneapi_bfloat16 = PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16,
};

enum class device_type : pi_uint64 {
  cpu = PI_DEVICE_TYPE_CPU,
  gpu = PI_DEVICE_TYPE_GPU,
  accelerator = PI_DEVICE_TYPE_ACC,
  // TODO: figure out if we need all the below in PI
  custom = PI_DEVICE_TYPE_CUSTOM,
  automatic,
  host,
  all = PI_DEVICE_TYPE_ALL
};

enum class partition_property : pi_device_partition_property {
  no_partition = 0,
  partition_equally = PI_DEVICE_PARTITION_EQUALLY,
  partition_by_counts = PI_DEVICE_PARTITION_BY_COUNTS,
  partition_by_affinity_domain = PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
};

enum class partition_affinity_domain : pi_device_affinity_domain {
  not_applicable = 0,
  numa = PI_DEVICE_AFFINITY_DOMAIN_NUMA,
  L4_cache = PI_DEVICE_AFFINITY_DOMAIN_L4_CACHE,
  L3_cache = PI_DEVICE_AFFINITY_DOMAIN_L3_CACHE,
  L2_cache = PI_DEVICE_AFFINITY_DOMAIN_L2_CACHE,
  L1_cache = PI_DEVICE_AFFINITY_DOMAIN_L1_CACHE,
  next_partitionable = PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : pi_device_fp_config {
  denorm = PI_FP_DENORM,
  inf_nan = PI_FP_INF_NAN,
  round_to_nearest = PI_FP_ROUND_TO_NEAREST,
  round_to_zero = PI_FP_ROUND_TO_ZERO,
  round_to_inf = PI_FP_ROUND_TO_INF,
  fma = PI_FP_FMA,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, read_write };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

// A.4 Queue information descriptors
enum class queue : pi_uint32 {
  context = PI_QUEUE_INFO_CONTEXT,
  device = PI_QUEUE_INFO_DEVICE,
  reference_count = PI_QUEUE_INFO_REFERENCE_COUNT
};

// A.5 Kernel information desctiptors
enum class kernel : pi_uint32 {
  function_name = PI_KERNEL_INFO_FUNCTION_NAME,
  num_args = PI_KERNEL_INFO_NUM_ARGS,
  context = PI_KERNEL_INFO_CONTEXT,
#ifdef __SYCL_INTERNAL_API
  program = PI_KERNEL_INFO_PROGRAM,
#endif
  reference_count = PI_KERNEL_INFO_REFERENCE_COUNT,
  attributes = PI_KERNEL_INFO_ATTRIBUTES
};

enum class __SYCL2020_DEPRECATED(
    "kernel_work_group enumeration is deprecated, use SYCL 2020 requests"
    " instead") kernel_work_group : pi_uint32 {
  global_work_size = PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE,
  work_group_size = PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
  compile_work_group_size = PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
  preferred_work_group_size_multiple =
      PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  private_mem_size = PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE
};

enum class kernel_sub_group : pi_uint32 {
  max_sub_group_size = PI_KERNEL_MAX_SUB_GROUP_SIZE,
  max_num_sub_groups = PI_KERNEL_MAX_NUM_SUB_GROUPS,
  compile_num_sub_groups = PI_KERNEL_COMPILE_NUM_SUB_GROUPS,
  compile_sub_group_size = PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
};

enum class kernel_device_specific : pi_uint32 {
  global_work_size = PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE,
  work_group_size = PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
  compile_work_group_size = PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
  preferred_work_group_size_multiple =
      PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  private_mem_size = PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE,
  ext_codeplay_num_regs = PI_KERNEL_GROUP_INFO_NUM_REGS,
  max_sub_group_size = PI_KERNEL_MAX_SUB_GROUP_SIZE,
  max_num_sub_groups = PI_KERNEL_MAX_NUM_SUB_GROUPS,
  compile_num_sub_groups = PI_KERNEL_COMPILE_NUM_SUB_GROUPS,
  compile_sub_group_size = PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
};

// A.6 Program information desctiptors
#ifdef __SYCL_INTERNAL_API
enum class program : pi_uint32 {
  context = PI_PROGRAM_INFO_CONTEXT,
  devices = PI_PROGRAM_INFO_DEVICES,
  reference_count = PI_PROGRAM_INFO_REFERENCE_COUNT
};
#endif

// A.7 Event information desctiptors
enum class event : pi_uint32 {
  reference_count = PI_EVENT_INFO_REFERENCE_COUNT,
  command_execution_status = PI_EVENT_INFO_COMMAND_EXECUTION_STATUS
};

enum class event_command_status : pi_int32 {
  submitted = PI_EVENT_SUBMITTED,
  running = PI_EVENT_RUNNING,
  complete = PI_EVENT_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};

enum class event_profiling : pi_uint32 {
  command_submit = PI_PROFILING_INFO_COMMAND_SUBMIT,
  command_start = PI_PROFILING_INFO_COMMAND_START,
  command_end = PI_PROFILING_INFO_COMMAND_END
};

// Provide an alias to the return type for each of the info parameters
template <typename T, T param> class param_traits {};

template <typename T, T param> struct compatibility_param_traits {};

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
  };

#define __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT(param_type, param, ret_type,       \
                                            in_type)                           \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
    using input_type = in_type;                                                \
  };

#include <sycl/info/device_traits.def>

#include <sycl/info/context_traits.def>

#include <sycl/info/event_traits.def>

#include <sycl/info/event_profiling_traits.def>

#include <sycl/info/kernel_device_specific_traits.def>
#include <sycl/info/kernel_sub_group_traits.def>
#include <sycl/info/kernel_traits.def>
#include <sycl/info/kernel_work_group_traits.def>

#include <sycl/info/platform_traits.def>

#ifdef __SYCL_INTERNAL_API
#include <sycl/info/program_traits.def>
#endif

#include <sycl/info/queue_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template <>                                                                  \
  struct compatibility_param_traits<param_type, param_type::param> {           \
    static constexpr auto value = kernel_device_specific::param;               \
  };

#define __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT(param_type, param, ret_type,       \
                                            in_type)                           \
  template <>                                                                  \
  struct compatibility_param_traits<param_type, param_type::param> {           \
    static constexpr auto value = kernel_device_specific::param;               \
  };

#include <sycl/info/kernel_sub_group_traits.def>
#include <sycl/info/kernel_work_group_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT

} // namespace info
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
