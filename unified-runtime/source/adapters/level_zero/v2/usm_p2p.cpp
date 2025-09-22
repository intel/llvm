//===----------- usm_p2p.cpp - L0 Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "logger/ur_logger.hpp"

namespace ur::level_zero {

static ur_result_t urUsmP2PChangePeerAccessExp(ur_device_handle_t commandDevice,
                                               ur_device_handle_t peerDevice,
                                               bool isAdding) {
  UR_LOG(INFO, "user tries to {} peer access to memory of {} from {}",
         (isAdding ? "enable" : "disable"), *peerDevice, *commandDevice);

  {
    const auto expectedPeerStatus =
        isAdding ? ur_device_handle_t_::PeerStatus::DISABLED
                 : ur_device_handle_t_::PeerStatus::ENABLED;
    std::shared_lock<ur_shared_mutex> Lock(commandDevice->Mutex);
    const auto existingPeerStatus =
        commandDevice->peers[peerDevice->Id.value()];
    if (existingPeerStatus != expectedPeerStatus) {
      UR_LOG(ERR,
             "existing peer status:{} does not match expected peer status:{}",
             existingPeerStatus, expectedPeerStatus);
      return UR_RESULT_ERROR_INVALID_OPERATION;
    }
    commandDevice->peers[peerDevice->Id.value()] =
        (isAdding ? ur_device_handle_t_::PeerStatus::ENABLED
                  : ur_device_handle_t_::PeerStatus::DISABLED);
  }

  auto Platform = commandDevice->Platform;
  {
    std::scoped_lock<ur_shared_mutex> Lock(Platform->ContextsMutex);
    UR_LOG(INFO, "changing peers in {} contexts", Platform->Contexts.size());
    for (auto Context : Platform->Contexts) {
      Context->changeResidentDevice(commandDevice, peerDevice, isAdding);
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urUsmP2PEnablePeerAccessExp(ur_device_handle_t commandDevice,
                                        ur_device_handle_t peerDevice) {
  return urUsmP2PChangePeerAccessExp(commandDevice, peerDevice, true);
}

ur_result_t urUsmP2PDisablePeerAccessExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice) {
  return urUsmP2PChangePeerAccessExp(commandDevice, peerDevice, false);
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
    std::scoped_lock<ur_shared_mutex> Lock(commandDevice->Mutex);
    propertyValue = commandDevice->peers[peerDevice->Id.value()] !=
                    ur_device_handle_t_::PeerStatus::NO_CONNECTION;
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
