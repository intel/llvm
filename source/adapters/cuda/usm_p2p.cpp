//===--------- usm_p2p.cpp - CUDA Adapter----------------------------------===//
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

  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(commandDevice->getContext());
    UR_CHECK_ERROR(cuCtxEnablePeerAccess(peerDevice->getContext(), 0));
  } catch (ur_result_t err) {
    result = err;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice) {

  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(commandDevice->getContext());
    UR_CHECK_ERROR(cuCtxDisablePeerAccess(peerDevice->getContext()));
  } catch (ur_result_t err) {
    result = err;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t commandDevice, ur_device_handle_t peerDevice,
    ur_exp_peer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  int value;
  CUdevice_P2PAttribute cuAttr;
  try {
    ScopedContext active(commandDevice->getContext());
    switch (propName) {
    case UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED: {
      cuAttr = CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED;
      break;
    }
    case UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED: {
      cuAttr = CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED;
      break;
    }
    default: {
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    }

    UR_CHECK_ERROR(cuDeviceGetP2PAttribute(&value, cuAttr, commandDevice->get(),
                                           peerDevice->get()));
  } catch (ur_result_t err) {
    return err;
  }
  return ReturnValue(value);
}
