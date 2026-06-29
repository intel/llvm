// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "unified-runtime/ur_api.h"

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
