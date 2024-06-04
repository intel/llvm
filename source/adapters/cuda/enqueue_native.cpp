//===--------- enqueue_native.cpp - CUDA Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "event.hpp"
#include "memory.hpp"
#include "queue.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t hQueue,
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t NumMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  std::vector<ur_event_handle_t> MemMigrationEvents;
  std::vector<std::pair<ur_mem_handle_t, ur_lock>> MemMigrationLocks;

  // phEventWaitList only contains events that are handed to UR by the SYCL
  // runtime. However since UR handles memory dependencies within a context
  // we may need to add more events to our dependent events list if the UR
  // context contains multiple devices
  if (NumMemsInMemList > 0 && hQueue->getContext()->Devices.size() > 1) {
    for (auto i = 0u; i < NumMemsInMemList; ++i) {
      auto Mem = phMemList[i];
      if (auto MemDepEvent = Mem->LastEventWritingToMemObj;
          MemDepEvent &&
          std::find(MemMigrationEvents.begin(), MemMigrationEvents.end(),
                    MemDepEvent) == MemMigrationEvents.end()) {
        MemMigrationEvents.push_back(MemDepEvent);
        MemMigrationLocks.emplace_back(
            std::pair{Mem, ur_lock{Mem->MemoryMigrationMutex}});
      }
    }
  }

  try {
    ScopedContext ActiveContext(hQueue->getDevice());
    ScopedStream ActiveStream(hQueue, NumEventsInWaitList, phEventWaitList);
    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

    if (phEvent || MemMigrationEvents.size()) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_ENQUEUE_NATIVE_EXP, hQueue, ActiveStream.getStream()));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    if (MemMigrationEvents.size()) {
      UR_CHECK_ERROR(
          urEnqueueEventsWaitWithBarrier(hQueue, MemMigrationEvents.size(),
                                         MemMigrationEvents.data(), nullptr));
      for (auto i = 0u; i < NumMemsInMemList; ++i) {
        auto Mem = phMemList[i];
        migrateMemoryToDeviceIfNeeded(Mem, hQueue->getDevice());
        Mem->setLastEventWritingToMemObj(RetImplEvent.get());
      }
      MemMigrationLocks.clear();
    }

    pfnNativeEnqueue(hQueue, data); // This is using urQueueGetNativeHandle to
                                    // get the CUDA stream. It must be the
                                    // same stream as is used before and after

    if (phEvent || MemMigrationEvents.size()) {
      UR_CHECK_ERROR(RetImplEvent->record());
      if (phEvent) {
        *phEvent = RetImplEvent.release();
      } else {
        // Give ownership of the event to the mem
        for (auto i = 0u; i < NumMemsInMemList; ++i) {
          auto Mem = phMemList[i];
          migrateMemoryToDeviceIfNeeded(Mem, hQueue->getDevice());
          Mem->setLastEventWritingToMemObj(RetImplEvent.release());
        }
      }
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (CUresult CuErr) {
    return mapErrorUR(CuErr);
  }
  return UR_RESULT_SUCCESS;
}
