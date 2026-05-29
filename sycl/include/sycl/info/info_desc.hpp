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

#define __SYCL_PARAM_TRAITS_DEPRECATED(Desc, Message)                          \
  struct __SYCL2020_DEPRECATED(Message) Desc;
#include <sycl/info/device_traits_2020_deprecated.def>
#undef __SYCL_PARAM_TRAITS_DEPRECATED

#define __SYCL_PARAM_TRAITS_DEPRECATED(Desc, Message)                          \
  struct __SYCL_DEPRECATED(Message) Desc;
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
#include <sycl/info/event_traits.def>
} // namespace event
namespace event_profiling {
#include <sycl/info/event_profiling_traits.def>
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
