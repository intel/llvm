//===--------- graph.cpp - HIP Adapter ------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL urGraphCreateExp(
    ur_context_handle_t /* hContext */, ur_exp_graph_handle_t * /* phGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphDestroyExp(ur_exp_graph_handle_t /* hGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphIsEmptyExp(ur_exp_graph_handle_t /* hGraph */, bool * /* pIsEmpty */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphGetIdExp(ur_exp_graph_handle_t /* hGraph */, uint64_t * /* pGraphId */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphSetDestructionCallbackExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_graph_destruction_callback_t /* pfnCallback */,
    void * /* pUserData */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphDumpContentsExp(
    ur_exp_graph_handle_t /* hGraph */, const char * /* pDotFilePath */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_executable_graph_handle_t * /* phExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGraphGetNativeHandleExp(ur_exp_graph_handle_t /* hGraph */,
                          ur_native_handle_t * /* phNativeGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGraphExecutableGraphGetNativeHandleExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */,
    ur_native_handle_t * /* phNativeExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
