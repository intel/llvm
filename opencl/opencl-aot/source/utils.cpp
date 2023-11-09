//===---- opencl-aot/source/utils.cpp - opencl-aot tool utils ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of support functions for opencl-aot
/// tool.
///
//===----------------------------------------------------------------------===//

#include "utils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <functional>

std::string getOpenCLErrorNameByErrorCode(cl_int CLErr) {
  switch (CLErr) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case CL_INVALID_PROPERTY:
    return "CL_INVALID_PROPERTY";
  case CL_INVALID_IMAGE_DESCRIPTOR:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case CL_INVALID_COMPILER_OPTIONS:
    return "CL_INVALID_COMPILER_OPTIONS";
  case CL_INVALID_LINKER_OPTIONS:
    return "CL_INVALID_LINKER_OPTIONS";
  case CL_INVALID_DEVICE_PARTITION_COUNT:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
#ifdef CL_VERSION_2_0
  case CL_INVALID_PIPE_SIZE:
    return "CL_INVALID_PIPE_SIZE";
  case CL_INVALID_DEVICE_QUEUE:
    return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
  case CL_INVALID_SPEC_ID:
    return "CL_INVALID_SPEC_ID";
  case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
    return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
  default:
    return "Unknown error code";
  }
}

std::tuple<cl_platform_id, std::string, std::string, cl_int>
getOpenCLPlatform(DeviceType Type) {
  const std::string StrIntelPlatformCommon("Intel(R) OpenCL");
  const std::string StrIntelPlatformNeoHd("Intel(R) OpenCL HD Graphics");
  const std::string StrIntelPlatformNeoUhd("Intel(R) OpenCL UHD Graphics");
  const std::string StrIntelPlatformCpuRt(
      "Intel(R) CPU Runtime for OpenCL(TM) Applications");
  const std::string StrIntelPlatformFastEmuPreview(
      "Intel(R) FPGA Emulation Platform for OpenCL(TM) (preview)");
  const std::string StrIntelPlatformFastEmu(
      "Intel(R) FPGA Emulation Platform for OpenCL(TM)");

  std::map<DeviceType, std::vector<std::string>>
      DeviceTypesToSupportedPlatformNames{
          {cpu, {StrIntelPlatformCommon, StrIntelPlatformCpuRt}},
          {gpu,
           {StrIntelPlatformCommon, StrIntelPlatformNeoHd,
            StrIntelPlatformNeoUhd}},
          {fpga_fast_emu,
           {StrIntelPlatformFastEmu, StrIntelPlatformFastEmuPreview}}};

  cl_platform_id PlatformId(nullptr);
  cl_int CLErr(CL_SUCCESS);
  std::string PlatformName;

  cl_uint PlatformsCount = 0;
  CLErr = clGetPlatformIDs(0, nullptr, &PlatformsCount);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        nullptr, "",
        formatCLError("Failed to retrieve OpenCL platform count", CLErr),
        CLErr);
  }

  std::vector<cl_platform_id> Platforms(PlatformsCount);
  CLErr = clGetPlatformIDs(PlatformsCount, Platforms.data(), nullptr);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        nullptr, "",
        formatCLError("Failed to retrieve OpenCL platform IDs", CLErr), CLErr);
  }

  std::string ErrorMessage;
  for (const auto &Platform : Platforms) {
    size_t PlatformNameLength = 0;
    CLErr = clGetPlatformInfo(Platform, CL_PLATFORM_NAME, 0, nullptr,
                              &PlatformNameLength);
    if (clFailed(CLErr)) {
      return std::make_tuple(
          nullptr, "",
          formatCLError("Failed to retrieve size of OpenCL platform name",
                        CLErr),
          CLErr);
    }

    std::string PlatformNameOnLoopIteration(
        PlatformNameLength - 1,
        '\0'); // std::string object adds another \0 to the end of the string,
               // so subtract 1 to avoid extra space in the of the string

    CLErr = clGetPlatformInfo(Platform, CL_PLATFORM_NAME, PlatformNameLength,
                              &PlatformNameOnLoopIteration.front(), nullptr);
    if (clFailed(CLErr)) {
      return std::make_tuple(
          nullptr, "",
          formatCLError("Failed to retrieve OpenCL platform name", CLErr),
          CLErr);
    }

    auto SupportedPlatformNames = DeviceTypesToSupportedPlatformNames[Type];
    auto Result =
        std::find(SupportedPlatformNames.begin(), SupportedPlatformNames.end(),
                  PlatformNameOnLoopIteration);
    if (Result != SupportedPlatformNames.end()) {
      tie(std::ignore, ErrorMessage, CLErr) = getOpenCLDevice(Platform, Type);
      if (!clFailed(CLErr)) {
        PlatformId = Platform;
        PlatformName = PlatformNameOnLoopIteration;
        break;
      }
    }
  }

  std::string SupportedPlatforms;
  for (const auto &Platform : DeviceTypesToSupportedPlatformNames[Type]) {
    SupportedPlatforms += "  " + Platform + '\n';
  }
  if (clFailed(CLErr)) {
    std::map<DeviceType, std::string> DeviceTypeToDeviceTypeName{
        {cpu, "CPU"}, {gpu, "GPU"}, {fpga_fast_emu, "FPGA Fast Emu"}};
    ErrorMessage += "Failed to find OpenCL " +
                    DeviceTypeToDeviceTypeName[Type] +
                    " device in these OpenCL platforms:\n" + SupportedPlatforms;
  } else {
    if (PlatformId == nullptr) {
      ErrorMessage += "OpenCL platform ID is empty\n";
    }
    if (PlatformName.empty()) {
      ErrorMessage += "OpenCL platform name is empty\n";
    }
    if (!ErrorMessage.empty()) {
      ErrorMessage += "Failed to find any of these OpenCL platforms:\n" +
                      SupportedPlatforms;
      CLErr = OPENCL_AOT_PLATFORM_NOT_FOUND;
    }
  }

  return std::make_tuple(PlatformId, PlatformName, ErrorMessage, CLErr);
}

std::tuple<cl_device_id, std::string, cl_int>
getOpenCLDevice(cl_platform_id PlatformId, DeviceType Type) {
  std::map<DeviceType, cl_device_type> DeviceTypesToOpenCLDeviceTypes{
      {cpu, CL_DEVICE_TYPE_CPU},
      {gpu, CL_DEVICE_TYPE_GPU},
      {fpga_fast_emu, CL_DEVICE_TYPE_ACCELERATOR}};

  cl_device_id DeviceId = nullptr;
  std::string ErrorMessage;
  cl_int CLErr = clGetDeviceIDs(
      PlatformId, DeviceTypesToOpenCLDeviceTypes[Type], 1, &DeviceId, nullptr);
  if (clFailed(CLErr)) {
    ErrorMessage += formatCLError("Failed to retrieve OpenCL device ID", CLErr);
  } else if (DeviceId == nullptr) {
    ErrorMessage += "OpenCL device ID is empty\n";
    CLErr = OPENCL_AOT_DEVICE_ID_IS_EMPTY;
  } // else is not needed

  return std::make_tuple(DeviceId, ErrorMessage, CLErr);
}

std::tuple<std::string, std::string, cl_int>
getOpenCLDeviceInfo(cl_device_id &DeviceId, cl_device_info ParamName) {
  size_t DeviceParameterSize = 0;

  cl_int CLErr =
      clGetDeviceInfo(DeviceId, ParamName, 0, nullptr, &DeviceParameterSize);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        "", formatCLError("Failed to get device parameter size", CLErr), CLErr);
  }

  std::string DeviceParameter(
      DeviceParameterSize - 1,
      '\0'); // std::string object adds another \0 to the end of the string, so
             // subtract 1 to avoid extra space in the of the string

  CLErr = clGetDeviceInfo(DeviceId, ParamName, DeviceParameterSize,
                          &DeviceParameter.front(), nullptr);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        "", formatCLError("Failed to get device parameter", CLErr), CLErr);
  }

  std::string ErrorMessage;
  if (DeviceParameter.empty()) {
    ErrorMessage += "OpenCL device parameter value is empty\n";
    CLErr = OPENCL_AOT_DEVICE_INFO_PARAMETER_IS_EMPTY;
  }
  return std::make_tuple(DeviceParameter, ErrorMessage, CLErr);
}

std::tuple<cl_program, std::string, cl_int>
createProgramWithIL(std::vector<char> IL, cl_context Context,
                    cl_platform_id PlatformId, cl_device_id DeviceId) {
  std::string DeviceVersionStr;
  std::string ErrorMessage;
  cl_int CLErr(CL_SUCCESS);
  cl_program Program(nullptr);
  std::tie(DeviceVersionStr, ErrorMessage, CLErr) =
      getOpenCLDeviceInfo(DeviceId, CL_DEVICE_VERSION);
  if (clFailed(CLErr)) {
    return std::make_tuple(Program, ErrorMessage, CLErr);
  }
  size_t BinarySize = IL.size();
  const unsigned char *BinaryContent = (const unsigned char *)IL.data();

  std::function<cl_program(cl_context, const void *, size_t, cl_int *)>
      CreateProgramWithIL;

  // according to OpenCL spec, the format of returned value is
  // OpenCL<space><major_version.minor_version><space><platform-specific
  // information>
  const char *DeviceVersion = &DeviceVersionStr[7]; // strlen("OpenCL ")
  if (DeviceVersion[0] == '2' &&
      DeviceVersion[2] > '0') // if OpenCL device version >= 2.1
  {
    CreateProgramWithIL = clCreateProgramWithIL;
  } else {
    std::string DeviceExtensions;
    std::tie(DeviceExtensions, ErrorMessage, CLErr) =
        getOpenCLDeviceInfo(DeviceId, CL_DEVICE_EXTENSIONS);
    if (clFailed(CLErr)) {
      return std::make_tuple(Program, ErrorMessage, CLErr);
    }
    if (DeviceExtensions.find("cl_khr_il_program") == std::string::npos) {
      ErrorMessage = "Unable to process SPIR-V binary! OpenCL "
                     "implementation version is less than 2.1 and doesn't "
                     "support cl_khr_il_program extension\n";
      return std::make_tuple(Program, ErrorMessage,
                             OPENCL_AOT_DEVICE_DOESNT_SUPPORT_SPIRV);
    }
    CreateProgramWithIL = reinterpret_cast<clCreateProgramWithILKHR_fn>(
        clGetExtensionFunctionAddressForPlatform(PlatformId,
                                                 "clCreateProgramWithILKHR"));
    if (!CreateProgramWithIL) {
      ErrorMessage =
          "clGetExtensionFunctionAddressForPlatform returned nullptr\n";
      return std::make_tuple(Program, ErrorMessage,
                             OPENCL_AOT_DEVICE_DOESNT_SUPPORT_SPIRV);
    }
  }
  Program = CreateProgramWithIL(Context, BinaryContent, BinarySize, &CLErr);
  if (Program == nullptr) {
    ErrorMessage = "OpenCL program is empty!";
    CLErr = OPENCL_AOT_PROGRAM_IS_EMPTY;
  }
  return std::make_tuple(Program, ErrorMessage, CLErr);
}

std::tuple<std::vector<char>, std::string, cl_int>
readBinaryFile(std::string FileName) {
  std::ifstream FileStream(FileName,
                           std::ios::in | std::ios::binary | std::ios::ate);
  if (!FileStream.is_open()) {
    return std::make_tuple(std::vector<char>(),
                           "Failed to open " + FileName + '\n',
                           OPENCL_AOT_FAILED_TO_OPEN_FILE);
  }
  std::ifstream::pos_type FileSize = FileStream.tellg();
  if (FileSize <= 0) { // it could be -1 if stream is corrupted
    std::string ErrorMessage = (FileSize == 0)
                                   ? FileName + " is empty\n"
                                   : "Failed to read " + FileName + '\n';
    return std::make_tuple(std::vector<char>(), ErrorMessage,
                           OPENCL_AOT_FILE_IS_EMPTY);
  }

  FileStream.seekg(0, std::ios::beg);

  std::vector<char> FileContent(FileSize);
  FileStream.read(&FileContent[0], FileSize);

  return std::make_tuple(FileContent, "", CL_SUCCESS);
}

bool isFileEndsWithGivenExtentionName(const std::string &FileName,
                                      const char *Ext) {
  std::size_t LastCharPosition = FileName.find_last_of('.');
  if (LastCharPosition == std::string::npos)
    return false;
  return FileName.substr(LastCharPosition) == Ext;
}

bool isFileStartsWithGivenMagicNumber(const std::vector<char> &BinaryData,
                                      const uint32_t ExpectedMagicNumber) {
  if (BinaryData.size() < sizeof(ExpectedMagicNumber))
    return false;
  const auto &BinaryDataAsIntBuffer =
      reinterpret_cast<decltype(ExpectedMagicNumber) *>(BinaryData.data());
  return BinaryDataAsIntBuffer[0] == ExpectedMagicNumber;
}

bool isFileOCLSource(const std::string &FileName) {
  return isFileEndsWithGivenExtentionName(FileName, ".cl");
}

bool isFileELF(const std::vector<char> &BinaryData) {
  const uint32_t ELFMagicNumber = 0x464c457f;
  return isFileStartsWithGivenMagicNumber(BinaryData, ELFMagicNumber);
}

bool isFileSPIRV(const std::vector<char> &BinaryData) {
  const uint32_t SPIRVMagicNumber = 0x07230203;
  return isFileStartsWithGivenMagicNumber(BinaryData, SPIRVMagicNumber);
}

bool isFileLLVMIR(const std::vector<char> &BinaryData) {
  const uint32_t LLVMIRMagicNumber = 0xdec04342;
  return isFileStartsWithGivenMagicNumber(BinaryData, LLVMIRMagicNumber);
}
