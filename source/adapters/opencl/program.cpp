//===--------- platform.cpp - OpenCL Adapter ---------------------------===//
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
#include "device.hpp"
#include "platform.hpp"

static ur_result_t getDevicesFromProgram(
    ur_program_handle_t hProgram,
    std::unique_ptr<std::vector<cl_device_id>> &DevicesInProgram) {

  cl_uint DeviceCount;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                        CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                                        &DeviceCount, nullptr));

  if (DeviceCount < 1) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  DevicesInProgram = std::make_unique<std::vector<cl_device_id>>(DeviceCount);

  CL_RETURN_ON_FAILURE(clGetProgramInfo(
      cl_adapter::cast<cl_program>(hProgram), CL_PROGRAM_DEVICES,
      DeviceCount * sizeof(cl_device_id), (*DevicesInProgram).data(), nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, const void *pIL, size_t length,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {

  std::unique_ptr<std::vector<cl_device_id>> DevicesInCtx;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      cl_adapter::getDevicesFromContext(hContext, DevicesInCtx), phProgram);

  cl_platform_id CurPlatform;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      clGetDeviceInfo((*DevicesInCtx)[0], CL_DEVICE_PLATFORM,
                      sizeof(cl_platform_id), &CurPlatform, nullptr),
      phProgram);

  oclv::OpenCLVersion PlatVer;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(
      cl_adapter::getPlatformVersion(CurPlatform, PlatVer), phProgram);

  cl_int Err = CL_SUCCESS;
  if (PlatVer >= oclv::V2_1) {

    /* Make sure all devices support CL 2.1 or newer as well. */
    for (cl_device_id Dev : *DevicesInCtx) {
      oclv::OpenCLVersion DevVer;

      CL_RETURN_ON_FAILURE_AND_SET_NULL(
          cl_adapter::getDeviceVersion(Dev, DevVer), phProgram);

      /* If the device does not support CL 2.1 or greater, we need to make sure
       * it supports the cl_khr_il_program extension.
       */
      if (DevVer < oclv::V2_1) {
        bool Supported = false;
        CL_RETURN_ON_FAILURE_AND_SET_NULL(
            cl_adapter::checkDeviceExtensions(Dev, {"cl_khr_il_program"},
                                              Supported),
            phProgram);

        if (!Supported) {
          return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
        }
      }
    }

    *phProgram = cl_adapter::cast<ur_program_handle_t>(clCreateProgramWithIL(
        cl_adapter::cast<cl_context>(hContext), pIL, length, &Err));
    CL_RETURN_ON_FAILURE(Err);
  } else {

    /* If none of the devices conform with CL 2.1 or newer make sure they all
     * support the cl_khr_il_program extension.
     */
    for (cl_device_id Dev : *DevicesInCtx) {
      bool Supported = false;
      CL_RETURN_ON_FAILURE_AND_SET_NULL(
          cl_adapter::checkDeviceExtensions(Dev, {"cl_khr_il_program"},
                                            Supported),
          phProgram);

      if (!Supported) {
        return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
      }
    }

    using ApiFuncT =
        cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
    ApiFuncT FuncPtr =
        reinterpret_cast<ApiFuncT>(clGetExtensionFunctionAddressForPlatform(
            CurPlatform, "clCreateProgramWithILKHR"));

    assert(FuncPtr != nullptr);

    *phProgram = cl_adapter::cast<ur_program_handle_t>(
        FuncPtr(cl_adapter::cast<cl_context>(hContext), pIL, length, &Err));
    CL_RETURN_ON_FAILURE(Err);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *,
    ur_program_handle_t *phProgram) {

  const cl_device_id Devices[1] = {cl_adapter::cast<cl_device_id>(hDevice)};
  const size_t Lengths[1] = {size};
  cl_int BinaryStatus[1];
  cl_int CLResult;
  *phProgram = cl_adapter::cast<ur_program_handle_t>(clCreateProgramWithBinary(
      cl_adapter::cast<cl_context>(hContext), cl_adapter::cast<cl_uint>(1u),
      Devices, Lengths, &pBinary, BinaryStatus, &CLResult));
  CL_RETURN_ON_FAILURE(BinaryStatus[0]);
  CL_RETURN_ON_FAILURE(CLResult);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile([[maybe_unused]] ur_context_handle_t hContext,
                 ur_program_handle_t hProgram, const char *pOptions) {

  std::unique_ptr<std::vector<cl_device_id>> DevicesInProgram;
  CL_RETURN_ON_FAILURE(getDevicesFromProgram(hProgram, DevicesInProgram));

  CL_RETURN_ON_FAILURE(clCompileProgram(cl_adapter::cast<cl_program>(hProgram),
                                        DevicesInProgram->size(),
                                        DevicesInProgram->data(), pOptions, 0,
                                        nullptr, nullptr, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

static cl_int mapURProgramInfoToCL(ur_program_info_t URPropName) {

  switch (static_cast<uint32_t>(URPropName)) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return CL_PROGRAM_REFERENCE_COUNT;
  case UR_PROGRAM_INFO_CONTEXT:
    return CL_PROGRAM_CONTEXT;
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return CL_PROGRAM_NUM_DEVICES;
  case UR_PROGRAM_INFO_DEVICES:
    return CL_PROGRAM_DEVICES;
  case UR_PROGRAM_INFO_SOURCE:
    return CL_PROGRAM_SOURCE;
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return CL_PROGRAM_BINARY_SIZES;
  case UR_PROGRAM_INFO_BINARIES:
    return CL_PROGRAM_BINARIES;
  case UR_PROGRAM_INFO_NUM_KERNELS:
    return CL_PROGRAM_NUM_KERNELS;
  case UR_PROGRAM_INFO_KERNEL_NAMES:
    return CL_PROGRAM_KERNEL_NAMES;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  size_t CheckPropSize = 0;
  auto ClResult = clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                   mapURProgramInfoToCL(propName), propSize,
                                   pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(ClResult);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuild([[maybe_unused]] ur_context_handle_t hContext,
               ur_program_handle_t hProgram, const char *pOptions) {

  std::unique_ptr<std::vector<cl_device_id>> DevicesInProgram;
  CL_RETURN_ON_FAILURE(getDevicesFromProgram(hProgram, DevicesInProgram));

  CL_RETURN_ON_FAILURE(clBuildProgram(
      cl_adapter::cast<cl_program>(hProgram), DevicesInProgram->size(),
      DevicesInProgram->data(), pOptions, nullptr, nullptr));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {

  cl_int CLResult;
  *phProgram = cl_adapter::cast<ur_program_handle_t>(
      clLinkProgram(cl_adapter::cast<cl_context>(hContext), 0, nullptr,
                    pOptions, cl_adapter::cast<cl_uint>(count),
                    cl_adapter::cast<const cl_program *>(phPrograms), nullptr,
                    nullptr, &CLResult));
  CL_RETURN_ON_FAILURE(CLResult);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompileExp(ur_program_handle_t,
                                                        uint32_t,
                                                        ur_device_handle_t *,
                                                        const char *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t,
                                                      uint32_t,
                                                      ur_device_handle_t *,
                                                      const char *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t, uint32_t, ur_device_handle_t *, uint32_t,
    const ur_program_handle_t *, const char *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

static cl_int mapURProgramBuildInfoToCL(ur_program_build_info_t URPropName) {

  switch (static_cast<uint32_t>(URPropName)) {
  case UR_PROGRAM_BUILD_INFO_STATUS:
    return CL_PROGRAM_BUILD_STATUS;
  case UR_PROGRAM_BUILD_INFO_OPTIONS:
    return CL_PROGRAM_BUILD_OPTIONS;
  case UR_PROGRAM_BUILD_INFO_LOG:
    return CL_PROGRAM_BUILD_LOG;
  case UR_PROGRAM_BUILD_INFO_BINARY_TYPE:
    return CL_PROGRAM_BINARY_TYPE;
  default:
    return -1;
  }
}

static ur_program_binary_type_t
mapCLBinaryTypeToUR(cl_program_binary_type binaryType) {
  switch (binaryType) {
  default:
    // If we don't understand what OpenCL gave us, return NONE.
    // TODO: Emit a warning to the user.
    [[fallthrough]];
  case CL_PROGRAM_BINARY_TYPE_INTERMEDIATE:
    // The INTERMEDIATE binary type is defined by the cl_khr_spir extension
    // which we shouldn't encounter but do. Semantically this binary type is
    // equivelent to NONE as they both require compilation.
    [[fallthrough]];
  case CL_PROGRAM_BINARY_TYPE_NONE:
    return UR_PROGRAM_BINARY_TYPE_NONE;
  case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
    return UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
  case CL_PROGRAM_BINARY_TYPE_LIBRARY:
    return UR_PROGRAM_BINARY_TYPE_LIBRARY;
  case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
    return UR_PROGRAM_BINARY_TYPE_EXECUTABLE;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t hDevice,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {
  if (propName == UR_PROGRAM_BUILD_INFO_BINARY_TYPE) {
    UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
    cl_program_binary_type BinaryType;
    CL_RETURN_ON_FAILURE(clGetProgramBuildInfo(
        cl_adapter::cast<cl_program>(hProgram),
        cl_adapter::cast<cl_device_id>(hDevice),
        mapURProgramBuildInfoToCL(propName), sizeof(cl_program_binary_type),
        &BinaryType, nullptr));
    return ReturnValue(mapCLBinaryTypeToUR(BinaryType));
  }
  size_t CheckPropSize = 0;
  cl_int ClErr = clGetProgramBuildInfo(cl_adapter::cast<cl_program>(hProgram),
                                       cl_adapter::cast<cl_device_id>(hDevice),
                                       mapURProgramBuildInfoToCL(propName),
                                       propSize, pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(ClErr);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {

  CL_RETURN_ON_FAILURE(clRetainProgram(cl_adapter::cast<cl_program>(hProgram)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {

  CL_RETURN_ON_FAILURE(
      clReleaseProgram(cl_adapter::cast<cl_program>(hProgram)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *phNativeProgram) {

  *phNativeProgram = reinterpret_cast<ur_native_handle_t>(hProgram);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t,
    const ur_program_native_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  *phProgram = reinterpret_cast<ur_program_handle_t>(hNativeProgram);
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urProgramRetain(*phProgram);
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {

  cl_program CLProg = cl_adapter::cast<cl_program>(hProgram);
  cl_context Ctx = nullptr;
  size_t RetSize = 0;

  CL_RETURN_ON_FAILURE(clGetProgramInfo(CLProg, CL_PROGRAM_CONTEXT, sizeof(Ctx),
                                        &Ctx, &RetSize));

  std::unique_ptr<std::vector<cl_device_id>> DevicesInCtx;
  UR_RETURN_ON_FAILURE(cl_adapter::getDevicesFromContext(
      cl_adapter::cast<ur_context_handle_t>(Ctx), DevicesInCtx));

  cl_platform_id CurPlatform;
  CL_RETURN_ON_FAILURE(clGetDeviceInfo((*DevicesInCtx)[0], CL_DEVICE_PLATFORM,
                                       sizeof(cl_platform_id), &CurPlatform,
                                       nullptr));

  oclv::OpenCLVersion PlatVer;
  cl_adapter::getPlatformVersion(CurPlatform, PlatVer);

  bool UseExtensionLookup = false;
  if (PlatVer < oclv::V2_2) {
    UseExtensionLookup = true;
  } else {
    for (cl_device_id Dev : *DevicesInCtx) {
      oclv::OpenCLVersion DevVer;

      UR_RETURN_ON_FAILURE(cl_adapter::getDeviceVersion(Dev, DevVer));

      if (DevVer < oclv::V2_2) {
        UseExtensionLookup = true;
        break;
      }
    }
  }

  if (UseExtensionLookup == false) {
    for (uint32_t i = 0; i < count; ++i) {
      CL_RETURN_ON_FAILURE(clSetProgramSpecializationConstant(
          CLProg, pSpecConstants[i].id, pSpecConstants[i].size,
          pSpecConstants[i].pValue));
    }
  } else {
    cl_ext::clSetProgramSpecializationConstant_fn
        SetProgramSpecializationConstant = nullptr;
    const ur_result_t URResult = cl_ext::getExtFuncFromContext<
        decltype(SetProgramSpecializationConstant)>(
        Ctx, cl_ext::ExtFuncPtrCache->clSetProgramSpecializationConstantCache,
        cl_ext::SetProgramSpecializationConstantName,
        &SetProgramSpecializationConstant);

    if (URResult != UR_RESULT_SUCCESS) {
      return URResult;
    }

    for (uint32_t i = 0; i < count; ++i) {
      CL_RETURN_ON_FAILURE(SetProgramSpecializationConstant(
          CLProg, pSpecConstants[i].id, pSpecConstants[i].size,
          pSpecConstants[i].pValue));
    }
  }
  return UR_RESULT_SUCCESS;
}

// Function gets characters between delimeter's in str
// then checks if they are equal to the sub_str.
// returns true if there is at least one instance
// returns false if there are no instances of the name
static bool isInSeparatedString(const std::string &Str, char Delimiter,
                                const std::string &SubStr) {
  size_t Beg = 0;
  size_t Length = 0;
  for (const auto &x : Str) {
    if (x == Delimiter) {
      if (Str.substr(Beg, Length) == SubStr)
        return true;

      Beg += Length + 1;
      Length = 0;
      continue;
    }
    Length++;
  }
  if (Length != 0)
    if (Str.substr(Beg, Length) == SubStr)
      return true;

  return false;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pFunctionName, void **ppFunctionPointer) {

  cl_context CLContext = nullptr;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                        CL_PROGRAM_CONTEXT, sizeof(CLContext),
                                        &CLContext, nullptr));

  cl_ext::clGetDeviceFunctionPointer_fn FuncT = nullptr;

  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clGetDeviceFunctionPointer_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clGetDeviceFunctionPointerCache,
          cl_ext::GetDeviceFunctionPointerName, &FuncT));

  if (!FuncT) {
    return UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
  }

  // Check if the kernel name exists to prevent the OpenCL runtime from throwing
  // an exception with the cpu runtime.
  // TODO: Use fallback search method if the clGetDeviceFunctionPointerINTEL
  // extension does not exist. Can only be done once the CPU runtime no longer
  // throws exceptions.
  *ppFunctionPointer = 0;
  size_t Size;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(cl_adapter::cast<cl_program>(hProgram),
                                        CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                        &Size));

  std::string KernelNames(Size, ' ');

  CL_RETURN_ON_FAILURE(clGetProgramInfo(
      cl_adapter::cast<cl_program>(hProgram), CL_PROGRAM_KERNEL_NAMES,
      KernelNames.size(), &KernelNames[0], nullptr));

  // Get rid of the null terminator and search for the kernel name. If the
  // function cannot be found, return an error code to indicate it exists.
  KernelNames.pop_back();
  if (!isInSeparatedString(KernelNames, ';', pFunctionName)) {
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  const cl_int CLResult =
      FuncT(cl_adapter::cast<cl_device_id>(hDevice),
            cl_adapter::cast<cl_program>(hProgram), pFunctionName,
            reinterpret_cast<cl_ulong *>(ppFunctionPointer));
  // GPU runtime sometimes returns CL_INVALID_ARG_VALUE if the function address
  // cannot be found but the kernel exists. As the kernel does exist, return
  // that the function name is invalid.
  if (CLResult == CL_INVALID_ARG_VALUE) {
    *ppFunctionPointer = 0;
    return UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
  }

  CL_RETURN_ON_FAILURE(CLResult);

  return UR_RESULT_SUCCESS;
}
