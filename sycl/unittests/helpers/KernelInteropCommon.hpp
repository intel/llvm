//==-- KernelInteropCommon.hpp --- Common kernel interop redefinitions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/UrMock.hpp>

struct TestContext {

  // SYCL RT has number of checks that all devices and contexts are consistent
  // between kernel, kernel_bundle and other objects.
  //
  // To ensure that those checks pass, we intercept some UR calls to extract
  // the exact UR handles of device and context used in queue creation to later
  // return them when program/context/kernel info is requested.
  ur_device_handle_t deviceHandle;
  ur_context_handle_t contextHandle;

  ur_program_handle_t programHandle =
      mock::createDummyHandle<ur_program_handle_t>();

  ~TestContext() {
    mock::releaseDummyHandle<ur_program_handle_t>(programHandle);
  }
};

TestContext GlobalContext;

ur_result_t after_urContextGetInfo(void *pParams) {
  auto params = *static_cast<ur_context_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_CONTEXT_INFO_DEVICES:
    if (*params.ppropName)
      *static_cast<ur_device_handle_t *>(*params.ppPropValue) =
          GlobalContext.deviceHandle;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(GlobalContext.deviceHandle);
    break;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t after_urProgramGetInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);

  switch (*params.ppropName) {
  case UR_PROGRAM_INFO_DEVICES:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(GlobalContext.deviceHandle);
    if (*params.ppPropValue)
      *static_cast<ur_device_handle_t *>(*params.ppPropValue) =
          GlobalContext.deviceHandle;
    break;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t redefined_urProgramGetBuildInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_build_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_PROGRAM_BUILD_INFO_BINARY_TYPE:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_program_binary_type_t);
    if (*params.ppPropValue)
      *static_cast<ur_program_binary_type_t *>(*params.ppPropValue) =
          UR_PROGRAM_BINARY_TYPE_EXECUTABLE;
    break;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t after_urContextCreate(void *pParams) {
  auto params = *static_cast<ur_context_create_params_t *>(pParams);
  if (*params.pphContext)
    GlobalContext.contextHandle = **params.pphContext;
  GlobalContext.deviceHandle = **params.pphDevices;
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_KERNEL_INFO_CONTEXT:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(GlobalContext.contextHandle);
    if (*params.ppPropValue)
      *static_cast<ur_context_handle_t *>(*params.ppPropValue) =
          GlobalContext.contextHandle;
    break;
  case UR_KERNEL_INFO_PROGRAM:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(GlobalContext.programHandle);
    if (*params.ppPropValue)
      *(ur_program_handle_t *)*params.ppPropValue = GlobalContext.programHandle;
    break;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

void redefineMockForKernelInterop(sycl::unittest::UrMock<> &Mock) {
  mock::getCallbacks().set_after_callback("urContextCreate",
                                          &after_urContextCreate);
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &after_urProgramGetInfo);
  mock::getCallbacks().set_after_callback("urContextGetInfo",
                                          &after_urContextGetInfo);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);
  mock::getCallbacks().set_replace_callback("urProgramGetBuildInfo",
                                            &redefined_urProgramGetBuildInfo);
}
