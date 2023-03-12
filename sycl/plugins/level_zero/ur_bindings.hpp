//===------ ur_bindings.hpp - Complete definitions of UR handles -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include "pi_level_zero.hpp"
#include <ur_api.h>

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
struct ur_platform_handle_t_ : public _pi_platform {
  using _pi_platform::_pi_platform;
};

struct ur_device_handle_t_ : public _pi_device {
  using _pi_device::_pi_device;
};
