//===--------- graph.cpp - OpenCL Adapter ---------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>

namespace ur::opencl {

ur_result_t urGraphCreateExp(ur_context_handle_t /* hContext */,
                             ur_exp_graph_handle_t * /* phGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphDestroyExp(ur_exp_graph_handle_t /* hGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphExecutableGraphDestroyExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphIsEmptyExp(ur_exp_graph_handle_t /* hGraph */,
                              bool * /* pIsEmpty */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphGetIdExp(ur_exp_graph_handle_t /* hGraph */,
                            uint64_t * /* pGraphId */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphSetDestructionCallbackExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_graph_destruction_callback_t /* pfnCallback */,
    void * /* pUserData */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphDumpContentsExp(ur_exp_graph_handle_t /* hGraph */,
                                   const char * /* pDotFilePath */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphInstantiateGraphExp(
    ur_exp_graph_handle_t /* hGraph */,
    ur_exp_executable_graph_handle_t * /* phExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urGraphGetNativeHandleExp(ur_exp_graph_handle_t /* hGraph */,
                          ur_native_handle_t * /* phNativeGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urGraphExecutableGraphGetNativeHandleExp(
    ur_exp_executable_graph_handle_t /* hExecutableGraph */,
    ur_native_handle_t * /* phNativeExecutableGraph */) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::opencl
