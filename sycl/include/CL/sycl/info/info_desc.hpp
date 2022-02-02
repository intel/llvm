//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/id.hpp>

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
enum class context : cl_context_info {
  reference_count = CL_CONTEXT_REFERENCE_COUNT,
  platform = CL_CONTEXT_PLATFORM,
  devices = CL_CONTEXT_DEVICES,
  atomic_memory_order_capabilities =
      PI_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
  atomic_memory_scope_capabilities =
      PI_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
};

// A.3 Device information descriptors
enum class device : cl_device_info {
  device_type = CL_DEVICE_TYPE,
  vendor_id = CL_DEVICE_VENDOR_ID,
  max_compute_units = CL_DEVICE_MAX_COMPUTE_UNITS,
  max_work_item_dimensions = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
  max_work_item_sizes = CL_DEVICE_MAX_WORK_ITEM_SIZES,
  max_work_group_size = CL_DEVICE_MAX_WORK_GROUP_SIZE,

  preferred_vector_width_char = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
  preferred_vector_width_short = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
  preferred_vector_width_int = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
  preferred_vector_width_long = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
  preferred_vector_width_float = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
  preferred_vector_width_double = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
  preferred_vector_width_half = CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,

  native_vector_width_char = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
  native_vector_width_short = CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
  native_vector_width_int = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
  native_vector_width_long = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
  native_vector_width_float = CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
  native_vector_width_double = CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
  native_vector_width_half = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,

  max_clock_frequency = CL_DEVICE_MAX_CLOCK_FREQUENCY,
  address_bits = CL_DEVICE_ADDRESS_BITS,
  max_mem_alloc_size = CL_DEVICE_MAX_MEM_ALLOC_SIZE,
  image_support = CL_DEVICE_IMAGE_SUPPORT,
  max_read_image_args = CL_DEVICE_MAX_READ_IMAGE_ARGS,
  max_write_image_args = CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
  image2d_max_width = CL_DEVICE_IMAGE2D_MAX_WIDTH,
  image2d_max_height = CL_DEVICE_IMAGE2D_MAX_HEIGHT,
  image3d_max_width = CL_DEVICE_IMAGE3D_MAX_WIDTH,
  image3d_max_height = CL_DEVICE_IMAGE3D_MAX_HEIGHT,
  image3d_max_depth = CL_DEVICE_IMAGE3D_MAX_DEPTH,
  image_max_buffer_size = CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
  image_max_array_size = CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
  max_samplers = CL_DEVICE_MAX_SAMPLERS,
  max_parameter_size = CL_DEVICE_MAX_PARAMETER_SIZE,
  mem_base_addr_align = CL_DEVICE_MEM_BASE_ADDR_ALIGN,
  half_fp_config = CL_DEVICE_HALF_FP_CONFIG,
  single_fp_config = CL_DEVICE_SINGLE_FP_CONFIG,
  double_fp_config = CL_DEVICE_DOUBLE_FP_CONFIG,
  global_mem_cache_type = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
  global_mem_cache_line_size = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
  global_mem_cache_size = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
  global_mem_size = CL_DEVICE_GLOBAL_MEM_SIZE,
  max_constant_buffer_size __SYCL2020_DEPRECATED(
      "max_constant_buffer_size is deprecated") =
      CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
  max_constant_args __SYCL2020_DEPRECATED("max_constant_args is deprecated") =
      CL_DEVICE_MAX_CONSTANT_ARGS,
  local_mem_type = CL_DEVICE_LOCAL_MEM_TYPE,
  local_mem_size = CL_DEVICE_LOCAL_MEM_SIZE,
  error_correction_support = CL_DEVICE_ERROR_CORRECTION_SUPPORT,
  host_unified_memory = CL_DEVICE_HOST_UNIFIED_MEMORY,
  profiling_timer_resolution = CL_DEVICE_PROFILING_TIMER_RESOLUTION,
  is_endian_little = CL_DEVICE_ENDIAN_LITTLE,
  is_available = CL_DEVICE_AVAILABLE,
  is_compiler_available = CL_DEVICE_COMPILER_AVAILABLE,
  is_linker_available = CL_DEVICE_LINKER_AVAILABLE,
  execution_capabilities = CL_DEVICE_EXECUTION_CAPABILITIES,
  queue_profiling = CL_DEVICE_QUEUE_PROPERTIES,
  built_in_kernels __SYCL2020_DEPRECATED("use built_in_kernel_ids instead") =
      CL_DEVICE_BUILT_IN_KERNELS,
  platform = CL_DEVICE_PLATFORM,
  name = CL_DEVICE_NAME,
  vendor = CL_DEVICE_VENDOR,
  driver_version = CL_DRIVER_VERSION,
  profile = CL_DEVICE_PROFILE,
  version = CL_DEVICE_VERSION,
  opencl_c_version = CL_DEVICE_OPENCL_C_VERSION,
  extensions __SYCL2020_DEPRECATED(
      "device::extensions is deprecated, use info::device::aspects"
      " instead.") = CL_DEVICE_EXTENSIONS,
  printf_buffer_size = CL_DEVICE_PRINTF_BUFFER_SIZE,
  preferred_interop_user_sync = CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
  parent_device = CL_DEVICE_PARENT_DEVICE,
  partition_max_sub_devices = CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
  partition_properties = CL_DEVICE_PARTITION_PROPERTIES,
  partition_affinity_domains = CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
  partition_type_affinity_domain = CL_DEVICE_PARTITION_TYPE,
  reference_count = CL_DEVICE_REFERENCE_COUNT,
  il_version =
      CL_DEVICE_IL_VERSION_KHR, // Same as CL_DEVICE_IL_VERSION for >=OpenCL 2.1
  max_num_sub_groups = CL_DEVICE_MAX_NUM_SUB_GROUPS,
  sub_group_independent_forward_progress =
      CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
  sub_group_sizes = CL_DEVICE_SUB_GROUP_SIZES_INTEL,
  partition_type_property,
  kernel_kernel_pipe_support,
  built_in_kernel_ids,
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
      PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES
};

enum class device_type : pi_uint64 {
  cpu         = PI_DEVICE_TYPE_CPU,
  gpu         = PI_DEVICE_TYPE_GPU,
  accelerator = PI_DEVICE_TYPE_ACC,
  // TODO: figure out if we need all the below in PI
  custom      = CL_DEVICE_TYPE_CUSTOM,
  automatic,
  host,
  all         = CL_DEVICE_TYPE_ALL
};

enum class partition_property : cl_device_partition_property {
  no_partition = 0,
  partition_equally = CL_DEVICE_PARTITION_EQUALLY,
  partition_by_counts = CL_DEVICE_PARTITION_BY_COUNTS,
  partition_by_affinity_domain = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
};

enum class partition_affinity_domain : cl_device_affinity_domain {
  not_applicable = 0,
  numa = CL_DEVICE_AFFINITY_DOMAIN_NUMA,
  L4_cache = CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE,
  L3_cache = CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE,
  L2_cache = CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE,
  L1_cache = CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE,
  next_partitionable = CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : cl_device_fp_config {
  denorm = CL_FP_DENORM,
  inf_nan = CL_FP_INF_NAN,
  round_to_nearest = CL_FP_ROUND_TO_NEAREST,
  round_to_zero = CL_FP_ROUND_TO_ZERO,
  round_to_inf = CL_FP_ROUND_TO_INF,
  fma = CL_FP_FMA,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, read_write };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

// A.4 Queue information descriptors
enum class queue : cl_command_queue_info {
  context = CL_QUEUE_CONTEXT,
  device = CL_QUEUE_DEVICE,
  reference_count = CL_QUEUE_REFERENCE_COUNT
};

// A.5 Kernel information desctiptors
enum class kernel : cl_kernel_info {
  function_name = CL_KERNEL_FUNCTION_NAME,
  num_args = CL_KERNEL_NUM_ARGS,
  context = CL_KERNEL_CONTEXT,
#ifdef __SYCL_INTERNAL_API
  program = CL_KERNEL_PROGRAM,
#endif
  reference_count = CL_KERNEL_REFERENCE_COUNT,
  attributes = CL_KERNEL_ATTRIBUTES
};

enum class __SYCL2020_DEPRECATED(
    "kernel_work_group enumeration is deprecated, use SYCL 2020 requests"
    " instead") kernel_work_group : cl_kernel_work_group_info {
  global_work_size = CL_KERNEL_GLOBAL_WORK_SIZE,
  work_group_size = CL_KERNEL_WORK_GROUP_SIZE,
  compile_work_group_size = CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
  preferred_work_group_size_multiple =
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  private_mem_size = CL_KERNEL_PRIVATE_MEM_SIZE
};

enum class kernel_sub_group : cl_kernel_sub_group_info {
  max_sub_group_size = CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
  max_num_sub_groups = CL_KERNEL_MAX_NUM_SUB_GROUPS,
  compile_num_sub_groups = CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
  compile_sub_group_size = CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
};

enum class kernel_device_specific : cl_kernel_work_group_info {
  global_work_size = CL_KERNEL_GLOBAL_WORK_SIZE,
  work_group_size = CL_KERNEL_WORK_GROUP_SIZE,
  compile_work_group_size = CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
  preferred_work_group_size_multiple =
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  private_mem_size = CL_KERNEL_PRIVATE_MEM_SIZE,
  ext_codeplay_num_regs = PI_KERNEL_GROUP_INFO_NUM_REGS,
  max_sub_group_size = CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
  max_num_sub_groups = CL_KERNEL_MAX_NUM_SUB_GROUPS,
  compile_num_sub_groups = CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
  compile_sub_group_size = CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
};

// A.6 Program information desctiptors
#ifdef __SYCL_INTERNAL_API
enum class program : cl_program_info {
  context = CL_PROGRAM_CONTEXT,
  devices = CL_PROGRAM_DEVICES,
  reference_count = CL_PROGRAM_REFERENCE_COUNT
};
#endif

// A.7 Event information desctiptors
enum class event : cl_event_info {
  reference_count = CL_EVENT_REFERENCE_COUNT,
  command_execution_status = CL_EVENT_COMMAND_EXECUTION_STATUS
};

enum class event_command_status : cl_int {
  submitted = CL_SUBMITTED,
  running = CL_RUNNING,
  complete = CL_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};

enum class event_profiling : cl_profiling_info {
  command_submit = CL_PROFILING_COMMAND_SUBMIT,
  command_start = CL_PROFILING_COMMAND_START,
  command_end = CL_PROFILING_COMMAND_END
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

#include <CL/sycl/info/device_traits.def>

#include <CL/sycl/info/context_traits.def>

#include <CL/sycl/info/event_traits.def>

#include <CL/sycl/info/event_profiling_traits.def>

#include <CL/sycl/info/kernel_device_specific_traits.def>
#include <CL/sycl/info/kernel_sub_group_traits.def>
#include <CL/sycl/info/kernel_traits.def>
#include <CL/sycl/info/kernel_work_group_traits.def>

#include <CL/sycl/info/platform_traits.def>

#ifdef __SYCL_INTERNAL_API
#include <CL/sycl/info/program_traits.def>
#endif

#include <CL/sycl/info/queue_traits.def>

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

#include <CL/sycl/info/kernel_sub_group_traits.def>
#include <CL/sycl/info/kernel_work_group_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT

} // namespace info
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
