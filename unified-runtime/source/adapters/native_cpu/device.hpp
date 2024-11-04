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

#include "threadpool.hpp"
#include <ur/ur.hpp>

struct ur_device_handle_t_ {
#ifdef NATIVECPU_WITH_ONETBB
  native_cpu::TBB_threadpool tp;
#else
  native_cpu::threadpool_t tp;
#endif
  ur_device_handle_t_(ur_platform_handle_t ArgPlt);

  const uint64_t mem_size;
  ur_platform_handle_t Platform;
};
