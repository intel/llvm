//===--------- graph.cpp - Level Zero Adapter -----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "graph.hpp"
#include "../external/driver_experimental/zex_graph.h"
#include "common.hpp"
#include "context.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v2 {

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext)
    : hContext(hContext) {
  auto ctx = v2_cast(hContext);
  ZE2UR_CALL_THROWS(ctx->getPlatform()->ZeGraphExt.zeGraphCreateExp,
                    (ctx->getZeHandle(), &zeGraph, nullptr));
}

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext,
                                               ze_graph_handle_t zeGraph)
    : hContext(hContext), zeGraph(zeGraph) {}

ur_exp_graph_handle_t_::~ur_exp_graph_handle_t_() {
  if (zeGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        v2_cast(hContext)->getPlatform()->ZeGraphExt.zeGraphDestroyExp,
        (zeGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG_SAFE(WARN, "Failed to destroy graph handle: {}", ZeResult);
    }
  }
}

ur_exp_executable_graph_handle_t_::ur_exp_executable_graph_handle_t_(
    ur_context_handle_t hContext, ur_exp_graph_handle_t hGraph)
    : hContext(hContext) {
  auto ctx = v2_cast(hContext);
  ZE2UR_CALL_THROWS(
      ctx->getPlatform()->ZeGraphExt.zeCommandListInstantiateGraphExp,
      (v2_cast(hGraph)->getZeHandle(), &zeExGraph, nullptr));
}

ur_exp_executable_graph_handle_t_::~ur_exp_executable_graph_handle_t_() {
  if (zeExGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        v2_cast(hContext)
            ->getPlatform()
            ->ZeGraphExt.zeExecutableGraphDestroyExp,
        (zeExGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG_SAFE(WARN, "Failed to destroy executable graph handle: {}",
                  ZeResult);
    }
  }
}

ur_result_t urGraphCreateExp(ur_context_handle_t hContext,
                             ur_exp_graph_handle_t *phGraph) try {
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto *pGraph = new ur_exp_graph_handle_t_(hContext);
  *phGraph = reinterpret_cast<::ur_exp_graph_handle_t>(pGraph);
  std::scoped_lock<ur_shared_mutex> lock(v2_cast(hContext)->GraphMapMutex);
  v2_cast(hContext)->registerGraph(pGraph->getZeHandle(), *phGraph);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t hGraph) try {
  auto *pGraph = v2_cast(hGraph);
  ur_context_handle_t hContext = pGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  {
    std::scoped_lock<ur_shared_mutex> lock(v2_cast(hContext)->GraphMapMutex);
    v2_cast(hContext)->unregisterGraph(pGraph->getZeHandle());
  }
  delete pGraph;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t hGraph,
    ur_exp_executable_graph_handle_t *phExecutableGraph) try {
  ur_context_handle_t hContext = v2_cast(hGraph)->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phExecutableGraph = reinterpret_cast<::ur_exp_executable_graph_handle_t>(
      new ur_exp_executable_graph_handle_t_(hContext, hGraph));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t hExecutableGraph) try {
  auto *pExec = v2_cast(hExecutableGraph);
  ur_context_handle_t hContext = pExec->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  delete pExec;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t hGraph, bool *pIsEmpty) {
  auto *pGraph = v2_cast(hGraph);
  ur_context_handle_t hContext = pGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ze_result_t zeResult =
      ZE_CALL_NOCHECK(v2_cast(hContext)->getPlatform()->ZeGraphExt.zeGraphIsEmptyExp,
                      (pGraph->getZeHandle()));
  if (zeResult == ZE_RESULT_ERROR_INVALID_GRAPH) {
    return UR_RESULT_ERROR_INVALID_GRAPH;
  }

  *pIsEmpty = (zeResult == ZE_RESULT_QUERY_TRUE);
  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphDumpContentsExp(ur_exp_graph_handle_t hGraph,
                                   const char *filePath) {
  auto *graph = v2_cast(hGraph);
  ur_context_handle_t hContext = graph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ZE2UR_CALL(
      v2_cast(hContext)->getPlatform()->ZeGraphExt.zeGraphDumpContentsExp,
      (graph->getZeHandle(), filePath, nullptr));

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero::v2
