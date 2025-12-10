//===--------- graph.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "graph.hpp"
#include "../external/driver_experimental/zex_graph.h"
#include "../ur_interface_loader.hpp"
#include "common.hpp"
#include "context.hpp"

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext)
    : hContext(hContext) {
  ZE2UR_CALL_THROWS(hContext->getPlatform()->ZeGraphExt.zeGraphCreateExp,
                    (hContext->getZeHandle(), &zeGraph, nullptr));
}

ur_exp_graph_handle_t_::ur_exp_graph_handle_t_(ur_context_handle_t hContext,
                                               ze_graph_handle_t zeGraph)
    : hContext(hContext), zeGraph(zeGraph) {}

ur_exp_graph_handle_t_::~ur_exp_graph_handle_t_() {
  if (zeGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        hContext->getPlatform()->ZeGraphExt.zeGraphDestroyExp, (zeGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG(WARN, "Failed to destroy graph handle: {}", ZeResult);
    }
  }
}

ur_exp_executable_graph_handle_t_::ur_exp_executable_graph_handle_t_(
    ur_context_handle_t hContext, ur_exp_graph_handle_t hGraph)
    : hContext(hContext) {
  ZE2UR_CALL_THROWS(
      hContext->getPlatform()->ZeGraphExt.zeCommandListInstantiateGraphExp,
      (hGraph->getZeHandle(), &zeExGraph, nullptr));
}

ur_exp_executable_graph_handle_t_::~ur_exp_executable_graph_handle_t_() {
  if (zeExGraph) {
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        hContext->getPlatform()->ZeGraphExt.zeExecutableGraphDestroyExp,
        (zeExGraph));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      UR_LOG(WARN, "Failed to destroy executable graph handle: {}", ZeResult);
    }
  }
}

namespace ur::level_zero {

ur_result_t urGraphCreateExp(ur_context_handle_t hContext,
                             ur_exp_graph_handle_t *phGraph) try {
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phGraph = new ur_exp_graph_handle_t_(hContext);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t hGraph) try {
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  delete hGraph;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t hGraph,
    ur_exp_executable_graph_handle_t *phExecutableGraph) try {
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  *phExecutableGraph = new ur_exp_executable_graph_handle_t_(hContext, hGraph);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t hExecutableGraph) try {
  ur_context_handle_t hContext = hExecutableGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  delete hExecutableGraph;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t hGraph, bool *pIsEmpty) {
  ur_context_handle_t hContext = hGraph->getContext();
  if (!checkGraphExtensionSupport(hContext)) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ze_result_t zeResult =
      ZE_CALL_NOCHECK(hContext->getPlatform()->ZeGraphExt.zeGraphIsEmptyExp,
                      (hGraph->getZeHandle()));
  if (zeResult == ZE_RESULT_ERROR_INVALID_GRAPH) {
    return UR_RESULT_ERROR_INVALID_GRAPH;
  }

  *pIsEmpty = (zeResult == ZE_RESULT_QUERY_TRUE);
  return UR_RESULT_SUCCESS;
}

ur_result_t urGraphDumpContentsExp(ur_exp_graph_handle_t, const char *) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
