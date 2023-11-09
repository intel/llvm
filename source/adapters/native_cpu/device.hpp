//===--------- device.hpp - Native CPU Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur/ur.hpp>

struct ur_device_handle_t_ {
  ur_device_handle_t_(ur_platform_handle_t ArgPlt) : Platform(ArgPlt) {}

  ur_platform_handle_t Platform;
};
