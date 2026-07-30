//===--------- device.hpp - Native CPU Adapter ----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include "threadpool.hpp"
#include <ur/ur.hpp>

struct ur_device_handle_t_ : ur::native_cpu::handle_base {
  native_cpu::threadpool_t tp;
  ur_device_handle_t_(ur_platform_handle_t ArgPlt);

  const uint64_t mem_size;
  ur_platform_handle_t Platform;
};
