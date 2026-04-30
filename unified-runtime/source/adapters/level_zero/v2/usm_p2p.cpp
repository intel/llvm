//===----------- usm_p2p.cpp - L0 Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device.hpp"
#include "context.hpp"
#include "logger/ur_logger.hpp"

namespace ur::level_zero {

// Validates that two devices are compatible for P2P operations: both must have
// an assigned Id, must belong to the same platform (i.e. share the same device
// cache), and peerDevice's id must be a valid index into commandDevice->peers.
static ur_result_t validateP2PDevicePair(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice) {
  if (!commandDevice || !peerDevice) {
    UR_LOG(ERR, "P2P operation requires non-null device handles");
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  if (!commandDevice->Id.has_value() || !peerDevice->Id.has_value()) {
    UR_LOG(ERR, "P2P operation requires devices with assigned ids");
    return UR_RESULT_ERROR_INVALID_DEVICE;
  }
  if (commandDevice->Platform != peerDevice->Platform) {
    UR_LOG(ERR, "P2P operation requires devices from the same platform");
    return UR_RESULT_ERROR_INVALID_DEVICE;
  }
  if (peerDevice->Id.value() >= commandDevice->peers.size()) {
    UR_LOG(ERR,
           "peerDevice id:{} is out of range for commandDevice peers table "
           "(size:{})",
           peerDevice->Id.value(), commandDevice->peers.size());
    return UR_RESULT_ERROR_INVALID_DEVICE;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t urUsmP2PChangePeerAccessExp(ur_device_handle_t commandDevice,
                                               ur_device_handle_t peerDevice,
                                               bool isAdding) {
  UR_CALL(validateP2PDevicePair(commandDevice, peerDevice));

  UR_LOG(INFO, "user tries to {} peer access to memory of {} from {}",
         (isAdding ? "enable" : "disable"), *peerDevice, *commandDevice);

  {
    const auto expectedPeerStatus =
        isAdding ? ur_device_handle_t_::PeerStatus::DISABLED
                 : ur_device_handle_t_::PeerStatus::ENABLED;
    std::scoped_lock<ur_shared_mutex> Lock(commandDevice->Mutex);
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
  // Copy the context list under the mutex and iterate outside the critical
  // section to avoid holding ContextsMutex during potentially heavy
  // changeResidentDevice calls and to reduce deadlock risk.
  std::list<ur_context_handle_t> Contexts;
  {
    std::scoped_lock<ur_shared_mutex> Lock(Platform->ContextsMutex);
    Contexts = Platform->Contexts;
  }
  UR_LOG(INFO, "changing peers in {} contexts", Contexts.size());
  for (auto Context : Contexts) {
    Context->changeResidentDevice(commandDevice, peerDevice, isAdding);
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

  UR_CALL(validateP2PDevicePair(commandDevice, peerDevice));

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
    propertyValue =
        (p2pProperties.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) != 0;
    break;
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return ReturnValue(propertyValue);
}
} // namespace ur::level_zero
