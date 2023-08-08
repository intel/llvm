//===--------- platform.hpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>
#include <vector>

struct ur_platform_handle_t_ {
  std::vector<std::unique_ptr<ur_device_handle_t_>> Devices;
};
