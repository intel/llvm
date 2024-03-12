//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/pi.h> // for PI_DEVICE_AFFINITY_DOMAIN_L...

// FIXME: .def files included to this file use all sorts of SYCL objects like
// id, range, traits, etc. We have to include some headers before including .def
// files.
#include <sycl/aspects.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>

#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {

class device;
class platform;
class kernel_id;
enum class memory_scope;
enum class memory_order;

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

struct atomic_fence_order_capabilities;
struct atomic_fence_scope_capabilities;

#define __SYCL_PARAM_TRAITS_DEPRECATED(Desc, Message)                          \
  struct __SYCL2020_DEPRECATED(Message) Desc;
#include <sycl/info/device_traits_deprecated.def>
#undef __SYCL_PARAM_TRAITS_DEPRECATED

template <int Dimensions = 3> struct max_work_item_sizes;
#define __SYCL_PARAM_TRAITS_TEMPLATE_SPEC(DescType, Desc, ReturnT, PiCode)     \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };
#define __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED(DescType, Desc, ReturnT, PiCode)  \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)

#include <sycl/info/device_traits.def>
} // namespace device
#undef __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED
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
#include <sycl/info/kernel_device_specific_traits.def>
} // namespace kernel_device_specific

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

namespace ext::oneapi::experimental::info::device {
template <int Dimensions> struct max_work_groups;
} // namespace ext::oneapi::experimental::info::device
#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC
} // namespace _V1
} // namespace sycl
