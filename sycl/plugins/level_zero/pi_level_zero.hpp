//===--------- pi_level_zero.hpp - Level Zero Plugin ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

/// \defgroup sycl_pi_level_zero Level Zero Plugin
/// \ingroup sycl_pi

/// \file pi_level_zero.hpp
/// Declarations for Level Zero Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying Level Zero runtime.
///
/// \ingroup sycl_pi_level_zero

#ifndef PI_LEVEL_ZERO_HPP
#define PI_LEVEL_ZERO_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_LEVEL_ZERO_PLUGIN_VERSION 1

#define _PI_LEVEL_ZERO_PLUGIN_VERSION_STRING                                   \
  _PI_PLUGIN_VERSION_STRING(_PI_LEVEL_ZERO_PLUGIN_VERSION)

// Share code between this PI L0 Plugin and UR L0 Adapter
#include <adapters/level_zero/ur_level_zero.hpp>
#include <pi2ur.hpp>

#endif // PI_LEVEL_ZERO_HPP
