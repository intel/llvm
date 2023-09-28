//===--------- usm_p2p.cpp - HIP Adapter-----------------------------------===//
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
urUsmP2PEnablePeerAccessExp(ur_device_handle_t, ur_device_handle_t) {
  detail::ur::die(
      "urUsmP2PEnablePeerAccessExp is not implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PDisablePeerAccessExp(ur_device_handle_t, ur_device_handle_t) {
  detail::ur::die(
      "urUsmP2PDisablePeerAccessExp is not implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t, ur_device_handle_t, ur_exp_peer_info_t, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  // Zero return value indicates that all of the queries currently return false.
  return ReturnValue(uint32_t{0});
}
