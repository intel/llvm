//===--------- platform.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"
#include "adapter.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "platform.hpp"

#include <vector>

ur_result_t ur_program_handle_t_::makeWithNative(native_type NativeProg,
                                                 ur_context_handle_t Context,
                                                 ur_program_handle_t &Program) {
  if (!Context) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  try {
    cl_context CLContext;
    CL_RETURN_ON_FAILURE(clGetProgramInfo(NativeProg, CL_PROGRAM_CONTEXT,
                                          sizeof(CLContext), &CLContext,
                                          nullptr));
    if (Context->CLContext != CLContext) {
      return UR_RESULT_ERROR_INVALID_CONTEXT;
    }
    auto URProgram = std::make_unique<ur_program_handle_t_>(
        NativeProg, Context, Context->DeviceCount, Context->Devices.data());
    Program = URProgram.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, const void *pIL, size_t length,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {

  ur_platform_handle_t CurPlatform = hContext->Devices[0]->Platform;

  oclv::OpenCLVersion PlatVer;
  CL_RETURN_ON_FAILURE_AND_SET_NULL(CurPlatform->getPlatformVersion(PlatVer),
                                    phProgram);

  cl_int Err = CL_SUCCESS;
  cl_program Program;
  if (PlatVer >= oclv::V2_1) {

    /* Make sure all devices support CL 2.1 or newer as well. */
    for (ur_device_handle_t URDev : hContext->Devices) {
      oclv::OpenCLVersion DevVer;

      CL_RETURN_ON_FAILURE_AND_SET_NULL(URDev->getDeviceVersion(DevVer),
                                        phProgram);

      /* If the device does not support CL 2.1 or greater, we need to make sure
       * it supports the cl_khr_il_program extension.
       */
      if (DevVer < oclv::V2_1) {
        bool Supported = false;
        CL_RETURN_ON_FAILURE_AND_SET_NULL(
            URDev->checkDeviceExtensions({"cl_khr_il_program"}, Supported),
            phProgram);

        if (!Supported) {
          return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
        }
      }
    }

    Program = clCreateProgramWithIL(hContext->CLContext, pIL, length, &Err);
  } else {
    /* If none of the devices conform with CL 2.1 or newer make sure they all
     * support the cl_khr_il_program extension.
     */
    for (ur_device_handle_t URDev : hContext->Devices) {
      bool Supported = false;
      CL_RETURN_ON_FAILURE_AND_SET_NULL(
          URDev->checkDeviceExtensions({"cl_khr_il_program"}, Supported),
          phProgram);

      if (!Supported) {
        return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
      }
    }

    cl_ext::clCreateProgramWithILKHR_fn CreateProgramWithIL = nullptr;

    UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext(
        hContext->CLContext,
        ur::cl::getAdapter()->fnCache.clCreateProgramWithILKHRCache,
        cl_ext::CreateProgramWithILName, &CreateProgramWithIL));

    Program = CreateProgramWithIL(hContext->CLContext, pIL, length, &Err);
  }

  // INVALID_VALUE is only returned in three circumstances according to the cl
  // spec:
  // * pIL == NULL
  // * length == 0
  // * pIL is not a well-formed binary
  // UR has a unique error code for each of these, so here we figure out which
  // to return
  if (Err == CL_INVALID_VALUE) {
    if (pIL == nullptr) {
      return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    if (length == 0) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    return UR_RESULT_ERROR_INVALID_BINARY;
  } else {
    CL_RETURN_ON_FAILURE(Err);
  }

  try {
    auto URProgram = std::make_unique<ur_program_handle_t_>(
        Program, hContext, hContext->DeviceCount, hContext->Devices.data());
    *phProgram = URProgram.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {
  std::vector<cl_device_id> CLDevices(numDevices);
  for (uint32_t i = 0; i < numDevices; ++i)
    CLDevices[i] = phDevices[i]->CLDevice;
  std::vector<cl_int> BinaryStatus(numDevices);
  cl_int CLResult;
  cl_program Program = clCreateProgramWithBinary(
      hContext->CLContext, static_cast<cl_uint>(numDevices), CLDevices.data(),
      pLengths, ppBinaries, BinaryStatus.data(), &CLResult);
  CL_RETURN_ON_FAILURE(CLResult);
  auto URProgram = std::make_unique<ur_program_handle_t_>(
      Program, hContext, numDevices, phDevices);
  *phProgram = URProgram.release();
  for (uint32_t i = 0; i < numDevices; ++i) {
    CL_RETURN_ON_FAILURE(BinaryStatus[i]);
  }
  CL_RETURN_ON_FAILURE(CLResult);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile([[maybe_unused]] ur_context_handle_t hContext,
                 ur_program_handle_t hProgram, const char *pOptions) {

  uint32_t DeviceCount = hProgram->NumDevices;
  std::vector<cl_device_id> CLDevicesInProgram(DeviceCount);
  for (uint32_t i = 0; i < DeviceCount; i++) {
    CLDevicesInProgram[i] = hProgram->Devices[i]->CLDevice;
  }

  CL_RETURN_ON_FAILURE(clCompileProgram(hProgram->CLProgram, DeviceCount,
                                        CLDevicesInProgram.data(), pOptions, 0,
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
  case UR_PROGRAM_INFO_IL:
    return CL_PROGRAM_IL;
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
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  const cl_program_info CLPropName = mapURProgramInfoToCL(propName);

  switch (static_cast<uint32_t>(propName)) {
  case UR_PROGRAM_INFO_CONTEXT: {
    return ReturnValue(hProgram->Context);
  }
  case UR_PROGRAM_INFO_NUM_DEVICES: {
    cl_uint DeviceCount = hProgram->NumDevices;
    return ReturnValue(DeviceCount);
  }
  case UR_PROGRAM_INFO_DEVICES: {
    return ReturnValue(hProgram->Devices.data(), hProgram->NumDevices);
  }
  case UR_PROGRAM_INFO_REFERENCE_COUNT: {
    return ReturnValue(hProgram->getReferenceCount());
  }
  default: {
    size_t CheckPropSize = 0;
    auto ClResult = clGetProgramInfo(hProgram->CLProgram, CLPropName, propSize,
                                     pPropValue, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    CL_RETURN_ON_FAILURE(ClResult);
    if (pPropSizeRet) {
      *pPropSizeRet = CheckPropSize;
    }
  }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuild([[maybe_unused]] ur_context_handle_t hContext,
               ur_program_handle_t hProgram, const char *pOptions) {

  uint32_t DeviceCount = hProgram->NumDevices;
  std::vector<cl_device_id> CLDevicesInProgram(DeviceCount);
  for (uint32_t i = 0; i < DeviceCount; i++) {
    CLDevicesInProgram[i] = hProgram->Devices[i]->CLDevice;
  }

  CL_RETURN_ON_FAILURE(
      clBuildProgram(hProgram->CLProgram, CLDevicesInProgram.size(),
                     CLDevicesInProgram.data(), pOptions, nullptr, nullptr));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {

  cl_int CLResult;
  std::vector<cl_program> CLPrograms(count);
  for (uint32_t i = 0; i < count; i++) {
    CLPrograms[i] = phPrograms[i]->CLProgram;
  }
  cl_program Program = clLinkProgram(
      hContext->CLContext, 0, nullptr, pOptions, static_cast<cl_uint>(count),
      CLPrograms.data(), nullptr, nullptr, &CLResult);

  if (CL_INVALID_BINARY == CLResult) {
    // Some OpenCL drivers incorrectly return CL_INVALID_BINARY here, convert it
    // to CL_LINK_PROGRAM_FAILURE
    CLResult = CL_LINK_PROGRAM_FAILURE;
  }
  CL_RETURN_ON_FAILURE(CLResult);
  try {
    auto URProgram = std::make_unique<ur_program_handle_t_>(
        Program, hContext, hContext->DeviceCount, hContext->Devices.data());
    *phProgram = URProgram.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

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
    const ur_program_handle_t *, const char *, ur_program_handle_t *phProgram) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }
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
        hProgram->CLProgram, hDevice->CLDevice,
        mapURProgramBuildInfoToCL(propName), sizeof(cl_program_binary_type),
        &BinaryType, nullptr));
    return ReturnValue(mapCLBinaryTypeToUR(BinaryType));
  }
  size_t CheckPropSize = 0;
  cl_int ClErr = clGetProgramBuildInfo(hProgram->CLProgram, hDevice->CLDevice,
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
  hProgram->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  if (hProgram->decrementReferenceCount() == 0) {
    delete hProgram;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *phNativeProgram) {

  *phNativeProgram = reinterpret_cast<ur_native_handle_t>(hProgram->CLProgram);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    const ur_program_native_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  cl_program NativeHandle = reinterpret_cast<cl_program>(hNativeProgram);

  UR_RETURN_ON_FAILURE(
      ur_program_handle_t_::makeWithNative(NativeHandle, hContext, *phProgram));
  (*phProgram)->IsNativeHandleOwned =
      pProperties ? pProperties->isNativeHandleOwned : false;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {

  cl_program CLProg = hProgram->CLProgram;
  if (!hProgram->Context) {
    return UR_RESULT_ERROR_INVALID_PROGRAM;
  }
  ur_context_handle_t Ctx = hProgram->Context;
  if (!Ctx->DeviceCount || !Ctx->Devices[0]->Platform) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  if (ur::cl::getAdapter()->clSetProgramSpecializationConstant) {
    for (uint32_t i = 0; i < count; ++i) {
      CL_RETURN_ON_FAILURE(
          ur::cl::getAdapter()->clSetProgramSpecializationConstant(
              CLProg, pSpecConstants[i].id, pSpecConstants[i].size,
              pSpecConstants[i].pValue));
    }
  } else {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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

  cl_context CLContext = hProgram->Context->CLContext;

  cl_ext::clGetDeviceFunctionPointer_fn FuncT = nullptr;

  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clGetDeviceFunctionPointer_fn>(
          CLContext,
          ur::cl::getAdapter()->fnCache.clGetDeviceFunctionPointerCache,
          cl_ext::GetDeviceFunctionPointerName, &FuncT));

  // Check if the kernel name exists to prevent the OpenCL runtime from throwing
  // an exception with the cpu runtime.
  // TODO: Use fallback search method if the clGetDeviceFunctionPointerINTEL
  // extension does not exist. Can only be done once the CPU runtime no longer
  // throws exceptions.
  *ppFunctionPointer = 0;
  size_t Size;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(
      hProgram->CLProgram, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &Size));

  std::string KernelNames(Size, ' ');

  CL_RETURN_ON_FAILURE(
      clGetProgramInfo(hProgram->CLProgram, CL_PROGRAM_KERNEL_NAMES,
                       KernelNames.size(), &KernelNames[0], nullptr));

  // Get rid of the null terminator and search for the kernel name. If the
  // function cannot be found, return an error code to indicate it exists.
  KernelNames.pop_back();
  if (!isInSeparatedString(KernelNames, ';', pFunctionName)) {
    return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
  }

  const cl_int CLResult =
      FuncT(hDevice->CLDevice, hProgram->CLProgram, pFunctionName,
            reinterpret_cast<cl_ulong *>(ppFunctionPointer));
  // GPU runtime sometimes returns CL_INVALID_ARG_VALUE if the function address
  // cannot be found but the kernel exists. As the kernel does exist, return
  // that the function name is invalid.
  if (CLResult == CL_INVALID_ARG_VALUE) {
    *ppFunctionPointer = 0;
    return UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE;
  }

  CL_RETURN_ON_FAILURE(CLResult);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetGlobalVariablePointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pGlobalVariableName, size_t *pGlobalVariableSizeRet,
    void **ppGlobalVariablePointerRet) {

  cl_context CLContext = nullptr;
  CL_RETURN_ON_FAILURE(clGetProgramInfo(hProgram->CLProgram, CL_PROGRAM_CONTEXT,
                                        sizeof(CLContext), &CLContext,
                                        nullptr));

  cl_ext::clGetDeviceGlobalVariablePointer_fn FuncT = nullptr;

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<
                       cl_ext::clGetDeviceGlobalVariablePointer_fn>(
      CLContext,
      ur::cl::getAdapter()->fnCache.clGetDeviceGlobalVariablePointerCache,
      cl_ext::GetDeviceGlobalVariablePointerName, &FuncT));

  const cl_int CLResult =
      FuncT(hDevice->CLDevice, hProgram->CLProgram, pGlobalVariableName,
            pGlobalVariableSizeRet, ppGlobalVariablePointerRet);

  if (CLResult != CL_SUCCESS) {
    *ppGlobalVariablePointerRet = nullptr;

    if (CLResult == CL_INVALID_ARG_VALUE) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    CL_RETURN_ON_FAILURE(CLResult);
  }

  return UR_RESULT_SUCCESS;
}
