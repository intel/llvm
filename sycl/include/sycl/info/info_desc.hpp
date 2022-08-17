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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

#ifdef __SYCL_INTERNAL_API
class program;
#endif
class device;
class platform;
class kernel_id;
enum class memory_scope;

// TODO: stop using OpenCL directly, use PI.
namespace info {
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  struct Desc {                                                                \
    using return_type = ReturnT;                                               \
  };
// A.1 Platform information desctiptors
namespace platform {
// TODO Despite giving this deprecation warning, we're still yet to implement
// info::device::aspects.
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::get_info() "
                             "with info::device::aspects instead") extensions;
#include <sycl/info/platform_traits.def>
} // namespace platform
// A.2 Context information desctiptors
namespace context {
#include <sycl/info/context_traits.def>
} // namespace context

// A.3 Device information descriptors
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

namespace device {
// TODO implement the following SYCL 2020 device info descriptors:
// atomic_fence_order_capabilities, atomic_fence_scope_capabilities, aspects,
// il_version.
// Marked deprecated in SYCL 2020 spec
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use device::has(aspect::image) instead")
    image_support;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
    max_constant_buffer_size;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") max_constant_args;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::has() with "
                             "one of the aspect::usm_* aspects instead")
    host_unified_memory;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, check the byte order of "
                             "the host system instead, the host and the device "
                             "are required to have the same byte order")
    is_endian_little;
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use device::has(aspect::online_compiler) instead")
    is_compiler_available;
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use device::has(aspect::online_linker) instead")
    is_linker_available;
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use device::has(aspect::queue_profiling) instead")
    queue_profiling;
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use info::device::built_in_kernel_ids instead")
    built_in_kernels;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") profile;
// TODO Despite giving this deprecation warning, we're still yet to implement
// info::device::aspects.
struct __SYCL2020_DEPRECATED(
    "deprecated in SYCL 2020, use info::device::aspects instead") extensions;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") printf_buffer_size;
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
    preferred_interop_user_sync;

// Deprecated and not part of SYCL 2020 spec
struct __SYCL2020_DEPRECATED("use info::device::usm_system_allocations instead")
    usm_system_allocator;

template <int Dimensions> struct max_work_item_sizes;
#define __SYCL_PARAM_TRAITS_TEMPLATE_SPEC(DescType, Desc, ReturnT, PiCode)     \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };
#include <sycl/info/device_traits.def>
} // namespace device
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC

// A.4 Queue information descriptors
namespace queue {
#include <sycl/info/queue_traits.def>
} // namespace queue

// A.5 Kernel information desctiptors
namespace kernel {
#include <sycl/info/kernel_traits.def>
} // namespace kernel

namespace kernel_device_specific {
#define __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT(DescType, Desc, ReturnT, InputT,   \
                                            PiCode)                            \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)
#include <sycl/info/kernel_device_specific_traits.def>
} // namespace kernel_device_specific
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT

// A.6 Event information desctiptors
enum class event_command_status : pi_int32 {
  submitted = PI_EVENT_SUBMITTED,
  running = PI_EVENT_RUNNING,
  complete = PI_EVENT_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};

namespace event {
#include <sycl/info/event_traits.def>
} // namespace event
namespace event_profiling {
#include <sycl/info/event_profiling_traits.def>
} // namespace event_profiling
#undef __SYCL_PARAM_TRAITS_SPEC

// Deprecated program class information desctiptors
#ifdef __SYCL_INTERNAL_API
enum class program : pi_uint32 {
  context = PI_PROGRAM_INFO_CONTEXT,
  devices = PI_PROGRAM_INFO_DEVICES,
  reference_count = PI_PROGRAM_INFO_REFERENCE_COUNT
};
#endif

// Provide an alias to the return type for each of the info parameters
template <typename T, T param> class param_traits {};

template <typename T, T param> struct compatibility_param_traits {};

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
  };

#ifdef __SYCL_INTERNAL_API
#include <sycl/info/program_traits.def>
#endif

#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace info
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
