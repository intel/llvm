//===--------- graph.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero {

// Graph experimental functions - not yet supported
ur_result_t urGraphCreateExp(ur_context_handle_t hContext,
                             ur_exp_graph_handle_t *phGraph) {
  std::ignore = hContext;
  if (phGraph)
    *phGraph = nullptr;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t hGraph) {
  std::ignore = hGraph;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t hExecutableGraph) {
  std::ignore = hExecutableGraph;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t hGraph, bool *pIsEmpty) {
  std::ignore = hGraph;
  if (pIsEmpty)
    *pIsEmpty = false;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphDumpContentsExp(ur_exp_graph_handle_t hGraph,
                                   const char *pDotFilePath) {
  std::ignore = hGraph;
  std::ignore = pDotFilePath;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t hGraph,
    ur_exp_executable_graph_handle_t *phExecutableGraph) {
  std::ignore = hGraph;
  if (phExecutableGraph)
    *phExecutableGraph = nullptr;
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
