//===--------- ipc_event.cpp - Level Zero Adapter -------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>

#include "../ur_interface_loader.hpp"

namespace ur::level_zero {

ur_result_t urIPCGetEventHandleExp(ur_event_handle_t /*hEvent*/,
                                   void ** /*ppIPCEventHandleData*/,
                                   size_t * /*pIPCEventHandleDataSizeRet*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCPutEventHandleExp(ur_context_handle_t /*hContext*/,
                                   void * /*pIPCEventHandleData*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCOpenEventHandleExp(ur_context_handle_t /*hContext*/,
                                    const void * /*pIPCEventHandleData*/,
                                    size_t /*ipcEventHandleDataSize*/,
                                    ur_event_handle_t * /*phEvent*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
