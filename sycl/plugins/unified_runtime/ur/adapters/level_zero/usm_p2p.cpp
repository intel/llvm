//===----------- usm_p2p.cpp - L0 Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice) {

  std::ignore = commandDevice;
  std::ignore = peerDevice;

  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice) {

  std::ignore = commandDevice;
  std::ignore = peerDevice;

  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice,
    ur_exp_peer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {

  std::ignore = commandDevice;
  std::ignore = peerDevice;
  std::ignore = propName;

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  // Zero return value indicates that all of the queries currently return false.
  return ReturnValue(uint32_t{0});
}
