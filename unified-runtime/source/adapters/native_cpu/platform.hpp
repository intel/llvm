//===--------- platform.hpp - Native CPU Adapter --------------------------===//
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

#include "common.hpp"
#include "device.hpp"

struct ur_platform_handle_t_ {
  ur_device_handle_t_ TheDevice{this};
};
