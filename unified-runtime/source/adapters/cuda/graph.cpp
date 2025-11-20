//===--------- graph.cpp - CUDA Adapter -----------------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

UR_APIEXPORT ur_result_t urGraphCreateExp(
    ur_context_handle_t /* hContext */, ur_exp_graph_handle_t * /* phGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t /* hGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t /* hGraph */,
                                           bool * /* pIsEmpty */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t urGraphDumpContentsExp(
    ur_exp_graph_handle_t /* hGraph */, const char * /* pDotFilePath */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_executable_graph_handle_t * /* phExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
