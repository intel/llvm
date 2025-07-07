// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "ur_api.h"

struct ur_queue_extensions {
  virtual ur_result_t
  onEventWaitListUse([[maybe_unused]] int64_t batch_generation) {
    return UR_RESULT_SUCCESS;
  }
};