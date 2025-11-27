//===--------- graph.cpp - Native CPU Adapter -----------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urGraphCreateExp(
    ur_context_handle_t /* hContext */, ur_exp_graph_handle_t * /* phGraph */) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphDestroyExp(ur_exp_graph_handle_t /* hGraph */) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphIsEmptyExp(ur_exp_graph_handle_t /* hGraph */, bool * /* pIsEmpty */) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphDumpContentsExp(
    ur_exp_graph_handle_t /* hGraph */, const char * /* pDotFilePath */) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_executable_graph_handle_t * /* phExecutableGraph */) {

  DIE_NO_IMPLEMENTATION;
}
