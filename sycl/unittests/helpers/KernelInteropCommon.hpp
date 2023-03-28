//==-- KernelInteropCommon.hpp --- Common kernel interop redefinitions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>

struct TestContext {

  // SYCL RT has number of checks that all devices and contexts are consistent
  // between kernel, kernel_bundle and other objects.
  //
  // To ensure that those checks pass, we intercept some PI calls to extract
  // the exact PI handles of device and context used in queue creation to later
  // return them when program/context/kernel info is requested.
  pi_device deviceHandle;
  pi_context contextHandle;

  pi_program programHandle = createDummyHandle<pi_program>();

  ~TestContext() { releaseDummyHandle<pi_program>(programHandle); }
};

TestContext GlobalContext;

pi_result after_piContextGetInfo(pi_context context, pi_context_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_DEVICES:
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalContext.deviceHandle;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.deviceHandle);
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result after_piProgramGetInfo(pi_program program, pi_program_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {

  switch (param_name) {
  case PI_PROGRAM_INFO_DEVICES:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.deviceHandle);
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalContext.deviceHandle;
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result redefined_piProgramGetBuildInfo(pi_program program, pi_device device,
                                          _pi_program_build_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_PROGRAM_BUILD_INFO_BINARY_TYPE:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_program_binary_type);
    if (param_value)
      *static_cast<pi_program_binary_type *>(param_value) =
          PI_PROGRAM_BINARY_TYPE_EXECUTABLE;
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result after_piContextCreate(const pi_context_properties *properties,
                                pi_uint32 num_devices, const pi_device *devices,
                                void (*pfn_notify)(const char *errinfo,
                                                   const void *private_info,
                                                   size_t cb, void *user_data),
                                void *user_data, pi_context *ret_context) {
  if (ret_context)
    GlobalContext.contextHandle = *ret_context;
  GlobalContext.deviceHandle = *devices;
  return PI_SUCCESS;
}

pi_result after_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_KERNEL_INFO_CONTEXT:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.contextHandle);
    if (param_value)
      *static_cast<pi_context *>(param_value) = GlobalContext.contextHandle;
    break;
  case PI_KERNEL_INFO_PROGRAM:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.programHandle);
    if (param_value)
      *(pi_program *)param_value = GlobalContext.programHandle;
    break;
  default:;
  }

  return PI_SUCCESS;
}

void redefineMockForKernelInterop(sycl::unittest::PiMock &Mock) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextCreate>(
      after_piContextCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piProgramGetInfo>(
      after_piProgramGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelGetInfo>(
      after_piKernelGetInfo);
  Mock.redefine<sycl::detail::PiApiKind::piProgramGetBuildInfo>(
      redefined_piProgramGetBuildInfo);
}
