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
#include "context.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice) {
  try {
    ScopedDevice active(commandDevice);
    UR_CHECK_ERROR(hipDeviceEnablePeerAccess(peerDevice->get(), 0));
  } catch (ur_result_t err) {
    return err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice) {
  try {
    ScopedDevice active(commandDevice);
    UR_CHECK_ERROR(hipDeviceDisablePeerAccess(peerDevice->get()));
  } catch (ur_result_t err) {
    return err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice,
    ur_exp_peer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  int value;
  hipDeviceP2PAttr hipAttr;
  try {
    ScopedDevice active(commandDevice);
    switch (propName) {
    case UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORT: {
      hipAttr = hipDevP2PAttrAccessSupported;
      break;
    }
    case UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORT: {
      hipAttr = hipDevP2PAttrNativeAtomicSupported;
      break;
    }
    default: {
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    }
    UR_CHECK_ERROR(hipDeviceGetP2PAttribute(
        &value, hipAttr, commandDevice->get(), peerDevice->get()));
  } catch (ur_result_t err) {
    return err;
  }
  return ReturnValue(value);
}
