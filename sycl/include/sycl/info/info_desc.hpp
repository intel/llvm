//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/pi.h>               // for PI_DEVICE_AFFINITY_DOMAIN_L...

#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED

namespace sycl {
inline namespace _V1 {


// TODO: stop using OpenCL directly, use PI.
namespace info {
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  struct Desc {                                                                \
    using return_type = ReturnT;                                               \
  };

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
  partition_by_affinity_domain = PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
  ext_intel_partition_by_cslice = PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE
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


#define __SYCL_PARAM_TRAITS_DEPRECATED(Desc, Message)                          \
  struct __SYCL2020_DEPRECATED(Message) Desc;
#undef __SYCL_PARAM_TRAITS_DEPRECATED

#define __SYCL_PARAM_TRAITS_TEMPLATE_SPEC(DescType, Desc, ReturnT, PiCode)     \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };
#define __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED(DescType, Desc, ReturnT, PiCode)  \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)

} // namespace device
#undef __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC

// A.6 Event information desctiptors
enum class event_command_status : pi_int32 {
  submitted = PI_EVENT_SUBMITTED,
  running = PI_EVENT_RUNNING,
  complete = PI_EVENT_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};
#undef __SYCL_PARAM_TRAITS_SPEC

// Provide an alias to the return type for each of the info parameters
template <typename T, T param> class param_traits {};

template <typename T, T param> struct compatibility_param_traits {};

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template <> class param_traits<param_type, param_type::param> {              \
  public:                                                                      \
    using return_type = ret_type;                                              \
  };
#undef __SYCL_PARAM_TRAITS_SPEC
} // namespace info

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, PiCode)   \
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
                                          PiCode)                              \
  namespace Namespace {                                                        \
  namespace info {                                                             \
  namespace DescType {                                                         \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };                                                                           \
  } /*namespace DescType */                                                    \
  } /*namespace info */                                                        \
  } /*namespace Namespace */

namespace ext::oneapi::experimental::info {

enum class graph_support_level { unsupported = 0, native, emulated };
} // namespace ext::oneapi::experimental::info
#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC
} // namespace _V1
} // namespace sycl
