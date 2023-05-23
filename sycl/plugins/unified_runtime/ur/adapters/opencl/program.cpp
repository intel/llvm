//===--------- platform.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "platform.hpp"

namespace cl {
cl_uint getDevicesFromProgram(
    ur_program_handle_t hProgram,
    std::unique_ptr<std::vector<cl_device_id>> &devicesInProgram) {

  cl_uint deviceCount;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                        CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                                        &deviceCount, nullptr));

  if (deviceCount < 1) {
    return CL_INVALID_CONTEXT;
  }

  devicesInProgram = std::make_unique<std::vector<cl_device_id>>(deviceCount);

  CL_RETURN_ON_FAILURE(clGetProgramInfo(
      cl_adapter::cast<cl_program>(hProgram), CL_PROGRAM_DEVICES,
      deviceCount * sizeof(cl_device_id), (*devicesInProgram).data(), nullptr));

  return CL_SUCCESS;
}
} // namespace cl

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, const void *pIL, size_t length,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pIL, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  std::unique_ptr<std::vector<cl_device_id>> devicesInCtx;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      cl_adapter::getDevicesFromContext(hContext, devicesInCtx), phProgram);

  cl_platform_id curPlatform;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      clGetDeviceInfo((*devicesInCtx)[0], CL_DEVICE_PLATFORM,
                      sizeof(cl_platform_id), &curPlatform, nullptr),
      phProgram);

  OCLV::OpenCLVersion platVer;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      cl::getPlatformVersion(curPlatform, platVer), phProgram);

  cl_int err = CL_SUCCESS;
  if (platVer >= OCLV::V2_1) {

    /* Make sure all devices support CL 2.1 or newer as well. */
    for (cl_device_id dev : *devicesInCtx) {
      OCLV::OpenCLVersion devVer;

      CL_RETURN_ON_FAILURE_AND_SET_NULL(cl_adapter::getDeviceVersion(dev, devVer),
                                        phProgram);

      /* If the device does not support CL 2.1 or greater, we need to make sure
       * it supports the cl_khr_il_program extension.
       */
      if (devVer < OCLV::V2_1) {
        bool supported = false;
        CL_RETURN_ON_FAILURE_AND_SET_NULL(
            cl_adapter::checkDeviceExtensions(dev, {"cl_khr_il_program"}, supported),
            phProgram);

        if (!supported) {
          return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
        }
      }
    }

    *phProgram = cl_adapter::cast<ur_program_handle_t>(clCreateProgramWithIL(
        cl_adapter::cast<cl_context>(hContext), pIL, length, &err));
    CL_RETURN_ON_FAILURE(err);
  }

  /* If none of the devices conform with CL 2.1 or newer make sure they all
   * support the cl_khr_il_program extension.
   */
  for (cl_device_id dev : *devicesInCtx) {
    bool supported = false;
    CL_RETURN_ON_FAILURE_AND_SET_NULL(
        cl_adapter::checkDeviceExtensions(dev, {"cl_khr_il_program"}, supported),
        phProgram);

    if (!supported) {
      return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
    }
  }

  using apiFuncT =
      cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
  apiFuncT funcPtr =
      reinterpret_cast<apiFuncT>(clGetExtensionFunctionAddressForPlatform(
          curPlatform, "clCreateProgramWithILKHR"));

  assert(funcPtr != nullptr);

  *phProgram = cl_adapter::cast<ur_program_handle_t>(
      funcPtr(cl_adapter::cast<cl_context>(hContext), pIL, length, &err));
  CL_RETURN_ON_FAILURE(err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *,
    ur_program_handle_t *phProgram) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pBinary, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int binary_status;
  cl_int cl_result;
  *phProgram = cl_adapter::cast<ur_program_handle_t>(clCreateProgramWithBinary(
      cl_adapter::cast<cl_context>(hContext), cl_adapter::cast<cl_uint>(1u),
      cl_adapter::cast<const cl_device_id *>(&hDevice), &size, &pBinary, &binary_status,
      &cl_result));
  CL_RETURN_ON_FAILURE(binary_status);
  CL_RETURN_ON_FAILURE(cl_result);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  std::unique_ptr<std::vector<cl_device_id>> devicesInProgram;
  CL_RETURN_ON_FAILURE(cl::getDevicesFromProgram(hProgram, devicesInProgram));

  CL_RETURN_ON_FAILURE(clCompileProgram(cl_adapter::cast<cl_program>(hProgram),
                                        devicesInProgram->size(),
                                        devicesInProgram->data(), pOptions, 0,
                                        nullptr, nullptr, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

cl_int map_ur_program_info_to_cl(ur_program_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    cl_propName = CL_PROGRAM_REFERENCE_COUNT;
    break;
  case UR_PROGRAM_INFO_CONTEXT:
    cl_propName = CL_PROGRAM_CONTEXT;
    break;
  case UR_PROGRAM_INFO_NUM_DEVICES:
    cl_propName = CL_PROGRAM_NUM_DEVICES;
    break;
  case UR_PROGRAM_INFO_DEVICES:
    cl_propName = CL_PROGRAM_DEVICES;
    break;
  case UR_PROGRAM_INFO_SOURCE:
    cl_propName = CL_PROGRAM_SOURCE;
    break;
  case UR_PROGRAM_INFO_BINARY_SIZES:
    cl_propName = CL_PROGRAM_BINARY_SIZES;
    break;
  case UR_PROGRAM_INFO_BINARIES:
    cl_propName = CL_PROGRAM_BINARIES;
    break;
  case UR_PROGRAM_INFO_NUM_KERNELS:
    cl_propName = CL_PROGRAM_NUM_KERNELS;
    break;
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    cl_propName = CL_PROGRAM_KERNEL_NAMES;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                        map_ur_program_info_to_cl(propName),
                                        propSize, pPropValue, pPropSizeRet));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t hContext,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  std::unique_ptr<std::vector<cl_device_id>> devicesInProgram;
  CL_RETURN_ON_FAILURE(cl::getDevicesFromProgram(hProgram, devicesInProgram));

  CL_RETURN_ON_FAILURE(
      clBuildProgram(cl_adapter::cast<cl_program>(hProgram), devicesInProgram->size(),
                     devicesInProgram->data(), pOptions, nullptr, nullptr));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phPrograms, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int cl_result;
  *phProgram = cl_adapter::cast<ur_program_handle_t>(clLinkProgram(
      cl_adapter::cast<cl_context>(hContext), 0, nullptr, pOptions,
      cl_adapter::cast<cl_uint>(count), cl_adapter::cast<const cl_program *>(phPrograms),
      nullptr, nullptr, &cl_result));
  CL_RETURN_ON_FAILURE(cl_result);

  return UR_RESULT_SUCCESS;
}

cl_int map_ur_program_build_info_to_cl(ur_program_build_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_PROGRAM_BUILD_INFO_STATUS:
    cl_propName = CL_PROGRAM_BUILD_STATUS;
    break;
  case UR_PROGRAM_BUILD_INFO_OPTIONS:
    cl_propName = CL_PROGRAM_BUILD_OPTIONS;
    break;
  case UR_PROGRAM_BUILD_INFO_LOG:
    cl_propName = CL_PROGRAM_BUILD_LOG;
    break;
  case UR_PROGRAM_BUILD_INFO_BINARY_TYPE:
    cl_propName = CL_PROGRAM_BINARY_TYPE;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t hDevice,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clGetProgramBuildInfo(
      cl_adapter::cast<cl_program>(hProgram), cl_adapter::cast<cl_device_id>(hDevice),
      map_ur_program_build_info_to_cl(propName), propSize, pPropValue,
      pPropSizeRet));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clRetainProgram(cl_adapter::cast<cl_program>(hProgram)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clReleaseProgram(cl_adapter::cast<cl_program>(hProgram)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *phNativeProgram) {

  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeProgram = reinterpret_cast<ur_native_handle_t>(hProgram);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t,
    const ur_program_native_properties_t *, ur_program_handle_t *phProgram) {
  UR_ASSERT(hNativeProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  *phProgram = reinterpret_cast<ur_program_handle_t>(hNativeProgram);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {

  cl_program clProg = cl_adapter::cast<cl_program>(hProgram);
  cl_context Ctx = nullptr;
  size_t RetSize = 0;

  CL_RETURN_ON_FAILURE(clGetProgramInfo(clProg, CL_PROGRAM_CONTEXT, sizeof(Ctx),
                                        &Ctx, &RetSize));

  cl_ext::clSetProgramSpecializationConstant_fn F = nullptr;
  const ur_result_t ur_result = cl_ext::getExtFuncFromContext<decltype(F)>(
      Ctx, cl_ext::ExtFuncPtrCache->clSetProgramSpecializationConstantCache,
      cl_ext::clSetProgramSpecializationConstantName, &F);

  if (ur_result != UR_RESULT_SUCCESS) {
    return ur_result;
  }

  for (uint32_t i = 0; i < count; ++i) {
    CL_RETURN_ON_FAILURE(F(clProg, pSpecConstants[i].id, pSpecConstants[i].size,
                           pSpecConstants[i].pValue));
  }

  return UR_RESULT_SUCCESS;
}
