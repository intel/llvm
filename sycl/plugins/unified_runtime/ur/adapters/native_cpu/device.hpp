//===--------- device.hpp - Native CPU Adapter 0----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#pragma once

#include <ur/ur.hpp>

struct ur_device_handle_t_ {
  ur_device_handle_t_(ur_platform_handle_t ArgPlt) : Platform(ArgPlt) {}

  ur_platform_handle_t Platform;
};
