//===--------- enqueue.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <ur_api.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t CommandQueue, CUstream Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList);
