//===--------- usm_p2p.cpp - OpenCL Adapter-------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PEnablePeerAccessExp([[maybe_unused]] ur_device_handle_t commandDevice,
                            [[maybe_unused]] ur_device_handle_t peerDevice) {

  die("Experimental P2P feature is not implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PDisablePeerAccessExp([[maybe_unused]] ur_device_handle_t commandDevice,
                             [[maybe_unused]] ur_device_handle_t peerDevice) {

  die("Experimental P2P feature is not implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    [[maybe_unused]] ur_device_handle_t commandDevice,
    [[maybe_unused]] ur_device_handle_t peerDevice,
    [[maybe_unused]] ur_exp_peer_info_t propName,
    [[maybe_unused]] size_t propSize, [[maybe_unused]] void *pPropValue,
    [[maybe_unused]] size_t *pPropSizeRet) {

  die("Experimental P2P feature is not implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
