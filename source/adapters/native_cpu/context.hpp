//===--------- context.hpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include "device.hpp"

struct ur_context_handle_t_ {
  ur_context_handle_t_(ur_device_handle_t_ *phDevices) : _device{phDevices} {}

  ur_device_handle_t _device;
};
