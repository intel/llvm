//==-- info_desc_traits.hpp - SYCL info descriptor self-describing traits --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

// Self-describing info-descriptor model.
//
// A trait struct describes itself via three nested members:
//   * `using return_type = ...;`     return type of `get_info<Trait>()`
//   * `using info_class  = ...;`     class tag (one of the structs below)
//   * `static constexpr auto ur_code = ...;`  UR enum or __SYCL_TRAIT_HANDLED_IN_RT
//
// This header replaces the .def-file + macro-iterator pattern used by
// `is_*_info_desc` / `UrInfoCode`. Both models coexist during migration; new
// traits should use the self-describing form, old traits will be migrated
// incrementally.

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace info_class {
struct platform {};
struct context {};
struct device {};
struct queue {};
struct kernel {};
struct kernel_device_specific {};
struct kernel_queue_specific {};
struct event {};
struct event_profiling {};
} // namespace info_class

template <typename T, typename = void>
struct is_self_describing_info_desc : std::false_type {};

template <typename T>
struct is_self_describing_info_desc<
    T, std::void_t<typename T::return_type, typename T::info_class,
                   decltype(T::ur_code)>> : std::true_type {};

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
