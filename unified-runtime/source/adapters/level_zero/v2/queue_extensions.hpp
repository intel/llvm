// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "ur_api.h"

struct ur_queue_extensions {
  // Non-batched queues don't need to perform any action
  //
  // This function is intended to be called by the event. If the event has been
  // created by the given queue and is associated with the current batch, this
  // batch should be enqueued for execution. Otherwise, the event would never be
  // signalled
  virtual ur_result_t
  onEventWaitListUse([[maybe_unused]] int64_t batch_generation) {
    return UR_RESULT_SUCCESS;
  }

  virtual bool isInOrder() = 0;
};
