//===----------- usm_p2p.cpp - L0 Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "logger/ur_logger.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero {

ur_result_t urUsmP2PEnablePeerAccessExp(ur_device_handle_t /*commandDevice*/,
                                        ur_device_handle_t /*peerDevice*/) {

  // L0 has peer devices enabled by default
  return UR_RESULT_SUCCESS;
}

ur_result_t urUsmP2PDisablePeerAccessExp(ur_device_handle_t /*commandDevice*/,
                                         ur_device_handle_t /*peerDevice*/) {

  // L0 has peer devices enabled by default
  return UR_RESULT_SUCCESS;
}

ur_result_t urUsmP2PPeerAccessGetInfoExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice,
                                         ur_exp_peer_info_t propName,
                                         size_t propSize, void *pPropValue,
                                         size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  int propertyValue = 0;
  switch (propName) {
  case UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORT: {
    bool p2pAccessSupported = false;
    ZeStruct<ze_device_p2p_properties_t> p2pProperties;
    ZE2UR_CALL(zeDeviceGetP2PProperties,
               (commandDevice->ZeDevice, peerDevice->ZeDevice, &p2pProperties));
    if (p2pProperties.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS) {
      p2pAccessSupported = true;
    }
    ze_bool_t p2pDeviceSupported = false;
    ZE2UR_CALL(
        zeDeviceCanAccessPeer,
        (commandDevice->ZeDevice, peerDevice->ZeDevice, &p2pDeviceSupported));
    propertyValue = p2pAccessSupported && p2pDeviceSupported;
    break;
  }
  case UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORT: {
    ZeStruct<ze_device_p2p_properties_t> p2pProperties;
    ZE2UR_CALL(zeDeviceGetP2PProperties,
               (commandDevice->ZeDevice, peerDevice->ZeDevice, &p2pProperties));
    propertyValue = p2pProperties.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS;
    break;
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return ReturnValue(propertyValue);
}
} // namespace ur::level_zero
