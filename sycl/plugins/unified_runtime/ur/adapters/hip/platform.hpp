//===--------- platform.hpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "device.hpp"

#include <vector>

/// A UR platform stores all known UR devices,
///  in the HIP plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct ur_platform_handle_t_ {
  static hipEvent_t evBase_; // HIP event used as base counter
  std::vector<std::unique_ptr<ur_device_handle_t_>> devices_;
};
