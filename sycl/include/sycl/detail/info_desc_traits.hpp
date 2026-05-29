//==-- info_desc_traits.hpp - SYCL info descriptor self-describing traits --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <unified-runtime/ur_api.h>

#include <type_traits>

// Self-describing info-descriptor model.
//
// A trait struct describes itself via these nested members:
//   * `using return_type = ...;`     return type of `get_info<Trait>()`
//   * `using info_class  = ...;`     class tag (one of the structs below)
//   * `static constexpr auto ur_code = ...;`  UR enum value
//
// `ur_code` is OPTIONAL. Traits dispatched via UR enum lookup must define it,
// of the family-specific UR enum type (see `ur_code_type` member of each
// info_class tag). Traits handled entirely SYCL-side (formerly tagged
// `__SYCL_TRAIT_HANDLED_IN_RT`) simply omit `ur_code`; runtime dispatch hits an
// explicit `CASE` switch in the *_impl headers and the UR fallthrough is never
// instantiated for them.
//
// This header replaces the .def-file + macro-iterator pattern used by
// `is_*_info_desc` / `UrInfoCode`. Both models coexist during migration; new
// traits should use the self-describing form, old traits will be migrated
// incrementally.

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace info_class {
struct platform {
  using ur_code_type = ur_platform_info_t;
};
struct context {
  using ur_code_type = ur_context_info_t;
};
struct device {
  using ur_code_type = ur_device_info_t;
};
struct queue {
  using ur_code_type = ur_queue_info_t;
};
struct kernel {
  using ur_code_type = ur_kernel_info_t;
};
struct kernel_device_specific {
  // kernel_device_specific traits dispatch through three different UR APIs
  // (urKernelGetGroupInfo, urKernelGetSubGroupInfo, urKernelGetInfo) and so
  // mix three native UR enum families. The runtime helper picks the right
  // call via IsSubGroupInfo / IsKernelInfo trait tags, passing
  // `UrInfoCode<T>::value` directly to the chosen API. We therefore allow
  // each trait to declare ur_code with its own native UR enum type and skip
  // the family static_assert by leaving ur_code_type as void.
  using ur_code_type = void;
};
struct kernel_queue_specific {
  // No UR enum lookup applies; queries dispatch through dedicated kernel/queue
  // call paths. Kept for type uniformity.
  using ur_code_type = void;
};
struct event {
  using ur_code_type = ur_event_info_t;
};
struct event_profiling {
  using ur_code_type = ur_profiling_info_t;
};
} // namespace info_class

template <typename T, typename = void>
struct is_self_describing_info_desc : std::false_type {};

template <typename T>
struct is_self_describing_info_desc<
    T, std::void_t<typename T::return_type, typename T::info_class>>
    : std::true_type {};

template <typename T, typename = void>
struct is_ur_dispatched : std::false_type {};

template <typename T>
struct is_ur_dispatched<T, std::void_t<decltype(T::ur_code)>>
    : std::true_type {};

template <typename T, typename Class, typename = void>
struct is_info_desc_for : std::false_type {};

template <typename T, typename Class>
struct is_info_desc_for<
    T, Class,
    std::enable_if_t<is_self_describing_info_desc<T>::value &&
                     std::is_same_v<typename T::info_class, Class>>>
    : std::true_type {
  using return_type = typename T::return_type;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
