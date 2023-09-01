//===--------- platform.hpp - Native CPU Adapter --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur/ur.hpp>

#include "common.hpp"
#include "device.hpp"

struct ur_platform_handle_t_ {
  ur_device_handle_t_ TheDevice{this};
};
