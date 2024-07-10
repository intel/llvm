//==---------- pi.hpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.hpp
/// C++ wrapper of extern "C" UR interfaces
///
/// \ingroup sycl_pi

#pragma once

#include <ur_api.h>

#include <sycl/backend_types.hpp>         // for backend
#include <sycl/detail/export.hpp>         // for __SYCL_EXPORT
#include <sycl/detail/os_util.hpp>        // for __SYCL_RT_OS_LINUX
#include <sycl/detail/ur_device_binary.h> // for pi binary stuff
                                          //
#include <memory>                         // for shared_ptr
#include <stddef.h>                       // for size_t
#include <string>                         // for char_traits, string
#include <vector>                         // for vector
namespace sycl {
inline namespace _V1 {

namespace detail {

namespace pi {} // namespace pi
} // namespace detail
} // namespace _V1
} // namespace sycl

#undef _PI_API
