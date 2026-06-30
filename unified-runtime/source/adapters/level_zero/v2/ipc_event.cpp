//===--------- ipc_event.cpp - Level Zero Adapter -------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>

#include <unified-runtime/ur_api.h>

#include "../platform.hpp"
#include "../ur_interface_loader.hpp"
#include "context.hpp"
#include "event.hpp"

namespace ur::level_zero {

namespace {

// Fixed L0 IPC handle size; reported by Get and validated by Open.
constexpr size_t kIpcEventHandleDataSize =
    sizeof(ze_ipc_event_counter_based_handle_t);

} // namespace

ur_result_t urIPCGetEventHandleExp(ur_event_handle_t hEvent,
                                   void **ppIPCEventHandleData,
                                   size_t *pIPCEventHandleDataSizeRet) try {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(ppIPCEventHandleData && pIPCEventHandleDataSizeRet,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  std::shared_lock<ur_shared_mutex> lock(hEvent->Mutex);

  UR_ASSERT(hEvent->isIpcCapable() && !hEvent->isIpcImported(),
            UR_RESULT_ERROR_INVALID_EVENT);
  UR_ASSERT(!hEvent->isProfilingEnabled() && !hEvent->isTimestamped(),
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  auto handle = std::make_unique<ze_ipc_event_counter_based_handle_t>();
  ZE2UR_CALL(zeEventCounterBasedGetIpcHandle,
             (hEvent->getZeEvent(), handle.get()));

  // Caller releases the buffer via urIPCPutEventHandleExp.
  *ppIPCEventHandleData = handle.release();
  *pIPCEventHandleDataSizeRet = kIpcEventHandleDataSize;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urIPCPutEventHandleExp(ur_context_handle_t /*hContext*/,
                                   void *pIPCEventHandleData) try {
  UR_ASSERT(pIPCEventHandleData, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // Free the buffer allocated by urIPCGetEventHandleExp via RAII.
  std::unique_ptr<ze_ipc_event_counter_based_handle_t> owner(
      static_cast<ze_ipc_event_counter_based_handle_t *>(pIPCEventHandleData));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urIPCOpenEventHandleExp(ur_context_handle_t hContext,
                                    const void *pIPCEventHandleData,
                                    size_t ipcEventHandleDataSize,
                                    ur_event_handle_t *phEvent) try {
  UR_ASSERT(pIPCEventHandleData && phEvent,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(ipcEventHandleDataSize == kIpcEventHandleDataSize,
            UR_RESULT_ERROR_INVALID_VALUE);

  // The driver consumes the handle by value.
  ze_ipc_event_counter_based_handle_t handle =
      *static_cast<const ze_ipc_event_counter_based_handle_t *>(
          pIPCEventHandleData);
  ze_event_handle_t hZeEvent = nullptr;
  ZE2UR_CALL(zeEventCounterBasedOpenIpcHandle,
             (hContext->getZeHandle(), handle, &hZeEvent));

  // The ipc_event_handle_t variant runs zeEventCounterBasedCloseIpcHandle on
  // urEventRelease.
  *phEvent =
      new ur_event_handle_t_(hContext, v2::raii::ipc_event_handle_t{hZeEvent},
                             v2::EVENT_FLAGS_COUNTER | v2::EVENT_FLAGS_IPC |
                                 v2::EVENT_FLAGS_IPC_IMPORTED);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

} // namespace ur::level_zero
