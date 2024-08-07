//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <ur_api.h>

// FIXME: .def files included to this file use all sorts of SYCL objects like
// id, range, traits, etc. We have to include some headers before including .def
// files.
#include <sycl/aspects.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>

#include <sycl/range.hpp>

// This is used in trait .def files when there isn't a corresponding backend
// query but we still need a value to instantiate the template.
#define __SYCL_TRAIT_HANDLED_IN_RT 0

namespace sycl {
inline namespace _V1 {

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
                             "with info::device::aspects instead") extensions;
#include <sycl/info/platform_traits.def>
} // namespace platform
// A.2 Context information desctiptors
namespace context {
#include <sycl/info/context_traits.def>
} // namespace context

// A.3 Device information descriptors
enum class device_type : uint32_t {
  cpu = UR_DEVICE_TYPE_CPU,
  gpu = UR_DEVICE_TYPE_GPU,
  accelerator = UR_DEVICE_TYPE_FPGA,
  // TODO: evaluate the need for equivalent UR enums for these types
  custom,
  automatic,
  host,
  all = UR_DEVICE_TYPE_ALL
};

enum class partition_property : intptr_t {
  no_partition = 0,
  partition_equally = UR_DEVICE_PARTITION_EQUALLY,
  partition_by_counts = UR_DEVICE_PARTITION_BY_COUNTS,
  partition_by_affinity_domain = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
  ext_intel_partition_by_cslice = UR_DEVICE_PARTITION_BY_CSLICE
};

// FIXME: maybe this should live elsewhere, maybe it should be implemented
// differently
inline partition_property
ConvertPartitionProperty(const ur_device_partition_t &Partition) {
  switch (Partition) {
  case UR_DEVICE_PARTITION_EQUALLY:
    return partition_property::partition_equally;
  case UR_DEVICE_PARTITION_BY_COUNTS:
    return partition_property::partition_by_counts;
  case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
    return partition_property::partition_by_affinity_domain;
  case UR_DEVICE_PARTITION_BY_CSLICE:
    return partition_property::ext_intel_partition_by_cslice;
  default:
    return partition_property::no_partition;
  }
}

enum class partition_affinity_domain : intptr_t {
  not_applicable = 0,
  numa = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
  L4_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE,
  L3_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE,
  L2_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE,
  L1_cache = UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE,
  next_partitionable = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE
};

inline partition_affinity_domain
ConvertAffinityDomain(const ur_device_affinity_domain_flags_t Domain) {
  switch (Domain) {
  case UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA:
    return partition_affinity_domain::numa;
  case UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE:
    return partition_affinity_domain::L1_cache;
  case UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE:
    return partition_affinity_domain::L2_cache;
  case UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE:
    return partition_affinity_domain::L3_cache;
  case UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE:
    return partition_affinity_domain::L4_cache;
  default:
    return info::partition_affinity_domain::not_applicable;
  }
}

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
#define __SYCL_PARAM_TRAITS_TEMPLATE_SPEC(DescType, Desc, ReturnT, UrCode)     \
  template <> struct Desc {                                                    \
    using return_type = ReturnT;                                               \
  };
#define __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED(DescType, Desc, ReturnT, UrCode)  \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)

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
enum class event_command_status : int32_t {
  submitted = UR_EVENT_STATUS_SUBMITTED,
  running = UR_EVENT_STATUS_RUNNING,
  complete = UR_EVENT_STATUS_COMPLETE,
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

namespace ext::oneapi::experimental::info::device {
template <int Dimensions> struct max_work_groups;
template <ext::oneapi::experimental::execution_scope CoordinationScope>
struct work_group_progress_capabilities;
template <ext::oneapi::experimental::execution_scope CoordinationScope>
struct sub_group_progress_capabilities;
template <ext::oneapi::experimental::execution_scope CoordinationScope>
struct work_item_progress_capabilities;

} // namespace ext::oneapi::experimental::info::device
#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_SPEC
} // namespace _V1
} // namespace sycl
