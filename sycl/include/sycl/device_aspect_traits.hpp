//==------------------- device_aspect_traits.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "device_aspect_macros.hpp" // for __SYCL_ALL_DEVICES_HAVE_* macro
#include <sycl/aspects.hpp>         // for aspect

#include <type_traits> // for bool_constant

// This macro creates an alias from an aspect to another. To avoid
// redeclarations, we need to define it empty for this file, otherwise we would
// have multiple declarations of `any_device_has` and `all_devices_have`, one
// for the original declaration of the aspect, and a subsequent one when finding
// the alias.
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)

namespace sycl {
template <aspect Aspect> struct all_devices_have;

// Macro to define `all_devices_have` traits for entries in aspects.def and
// aspects_deprecated.def.
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  template <>                                                                  \
  struct all_devices_have<aspect::ASPECT>                                      \
      : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_##ASPECT##__> {};

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  template <>                                                                  \
  struct all_devices_have<aspect::ASPECT>                                      \
      : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_##ASPECT##__> {};
#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED

#ifdef __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__
// Special case where `any_device_has` is trivially true.
template <aspect Aspect> struct any_device_has : std::true_type {};
#else
template <aspect Aspect> struct any_device_has;

// Macro to define `any_device_has` traits for entries in aspects.def and
// aspects_deprecated.def.
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  template <>                                                                  \
  struct any_device_has<aspect::ASPECT>                                        \
      : std::bool_constant<__SYCL_ANY_DEVICE_HAS_##ASPECT##__> {};

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  template <>                                                                  \
  struct any_device_has<aspect::ASPECT>                                        \
      : std::bool_constant<__SYCL_ANY_DEVICE_HAS_##ASPECT##__> {};
#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED
#endif // __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__

template <aspect Aspect>
constexpr bool all_devices_have_v = all_devices_have<Aspect>::value;
template <aspect Aspect>
constexpr bool any_device_has_v = any_device_has<Aspect>::value;
} // namespace sycl

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
