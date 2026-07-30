//===--------- graph.cpp - Level Zero Adapter -----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "graph.hpp"
#include "common.hpp"
#include "context.hpp"
#include "ur_interface_loader.hpp"

#include <memory>

namespace ur::level_zero::v2 {

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext)
    : hContext(hContext) {
  ZE2UR_CALL_THROWS(hContext->getPlatform()->ZeGraphExt.graphCreate,
                    (hContext->getZeHandle(), nullptr, &zeGraph));
}

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext,
                                               ze_graph_handle_t zeGraph)
    : hContext(hContext), zeGraph(zeGraph) {}

ur_exp_graph_handle_t_::~ur_exp_graph_handle_t_() {
  if (zeGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        hContext->getPlatform()->ZeGraphExt.zeGraphDestroyExp, (zeGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG_SAFE(WARN, "Failed to destroy graph handle: {}", ZeResult);
    }
  }
}

ur_exp_executable_graph_handle_t_::ur_exp_executable_graph_handle_t_(
    ur_context_handle_t hContext, ur_exp_graph_handle_t hGraph)
    : hContext(hContext) {
  ZE2UR_CALL_THROWS(hContext->getPlatform()->ZeGraphExt.instantiateGraph,
                    (hGraph->getZeHandle(), nullptr, &zeExGraph));
}

ur_exp_executable_graph_handle_t_::~ur_exp_executable_graph_handle_t_() {
  if (zeExGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        hContext->getPlatform()->ZeGraphExt.zeExecutableGraphDestroyExp,
        (zeExGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG_SAFE(WARN, "Failed to destroy executable graph handle: {}",
                  ZeResult);
    }
  }
}

namespace {

// L0 imposes specific callback conventions. We must wrap the UR callback in
// order to not violate this requirement.
struct DestructionCallbackContext {
  ur_exp_graph_destruction_callback_t callback;
  void *userData;
};

// Must match the calling convention of zeGraphSetDestructionCallbackExt's
// pfnCallback parameter, declared as zex_mem_graph_free_callback_fn_t in
// ze_api.h.
void ZE_CALLBACK_CONV destructionCallbackWrapper(void *pUserData) {
  auto *CbData = static_cast<DestructionCallbackContext *>(pUserData);
  CbData->callback(CbData->userData);
  delete CbData;
}

} // namespace

ur_result_t urGraphCreateExp(::ur_context_handle_t hContextOpque,
                             ::ur_exp_graph_handle_t *phGraphOpque) try {
  auto hContext = v2_cast(hContextOpque);
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto hGraph = new ur_exp_graph_handle_t_(hContext);
  *phGraphOpque = v2_cast(hGraph);
  std::scoped_lock<ur_shared_mutex> lock(hContext->GraphMapMutex);
  hContext->registerGraph(hGraph->getZeHandle(), hGraph);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphDestroyExp(::ur_exp_graph_handle_t hGraphOpque) try {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  {
    std::scoped_lock<ur_shared_mutex> lock(hContext->GraphMapMutex);
    hContext->unregisterGraph(hGraph->getZeHandle());
  }
  delete hGraph;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphInstantiateGraphExp(
    ::ur_exp_graph_handle_t hGraphOpque,
    ::ur_exp_executable_graph_handle_t *phExecutableGraphOpque) try {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phExecutableGraphOpque =
      v2_cast(new ur_exp_executable_graph_handle_t_(hContext, hGraph));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphExecutableGraphDestroyExp(
    ::ur_exp_executable_graph_handle_t hExecutableGraphOpque) try {
  auto hExecutableGraph = v2_cast(hExecutableGraphOpque);
  ur_context_handle_t hContext = hExecutableGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  delete hExecutableGraph;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphIsEmptyExp(::ur_exp_graph_handle_t hGraphOpque,
                              bool *pIsEmpty) {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto &ZeGraphExt = hContext->getPlatform()->ZeGraphExt;
  ze_result_t zeResult = ZeGraphExt.normalizeGraphQueryResult(
      ZE_CALL_NOCHECK(ZeGraphExt.zeGraphIsEmptyExp, (hGraph->getZeHandle())));
  if (zeResult == ZE_RESULT_ERROR_INVALID_GRAPH) {
    return UR_RESULT_ERROR_INVALID_GRAPH;
  }

  *pIsEmpty = (zeResult == ZE_RESULT_QUERY_TRUE);
  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphGetIdExp(::ur_exp_graph_handle_t hGraphOpque,
                            uint64_t *pGraphId) {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  auto ZeGetId = hContext->getPlatform()->ZeGraphExt.zeGraphGetIdExt;
  if (!checkGraphExtensionSupport(hContext) || !ZeGetId) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(ZeGetId, (hGraph->getZeHandle(), pGraphId));

  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphSetDestructionCallbackExp(
    ::ur_exp_graph_handle_t hGraphOpque,
    ur_exp_graph_destruction_callback_t pfnCallback, void *pUserData) {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  auto ZeSetCallback =
      hContext->getPlatform()->ZeGraphExt.zeGraphSetDestructionCallbackExp;
  if (!checkGraphExtensionSupport(hContext) || !ZeSetCallback) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto CbData = std::make_unique<DestructionCallbackContext>(
      DestructionCallbackContext{pfnCallback, pUserData});

  ze_result_t ZeResult = ZE_CALL_NOCHECK(
      ZeSetCallback, (hGraph->getZeHandle(), destructionCallbackWrapper,
                      static_cast<void *>(CbData.get()), nullptr));

  if (ZeResult != ZE_RESULT_SUCCESS) {
    return ze2urResult(ZeResult);
  }

  // Ownership is transfered to the graph destruction callback for deletion
  CbData.release();
  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphDumpContentsExp(::ur_exp_graph_handle_t hGraphOpque,
                                   const char *filePath) {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(hContext->getPlatform()->ZeGraphExt.zeGraphDumpContentsExp,
             (hGraph->getZeHandle(), filePath, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphGetNativeHandleExp(::ur_exp_graph_handle_t hGraphOpque,
                                      ur_native_handle_t *phNativeGraph) try {
  auto hGraph = v2_cast(hGraphOpque);
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phNativeGraph = reinterpret_cast<ur_native_handle_t>(hGraph->getZeHandle());
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphExecutableGraphGetNativeHandleExp(
    ::ur_exp_executable_graph_handle_t hExecutableGraphOpque,
    ur_native_handle_t *phNativeExecutableGraph) try {
  auto hExecutableGraph = v2_cast(hExecutableGraphOpque);
  ur_context_handle_t hContext = hExecutableGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phNativeExecutableGraph =
      reinterpret_cast<ur_native_handle_t>(hExecutableGraph->getZeHandle());
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

} // namespace ur::level_zero::v2
