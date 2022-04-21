//===---- opencl-aot/include/utils.h - opencl-aot tool utils ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of support functions for opencl-aot tool.
///
//===----------------------------------------------------------------------===//

#ifndef AOT_COMP_TOOL_UTILS
#define AOT_COMP_TOOL_UTILS

#include <CL/cl_ext.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

enum DeviceType : int8_t { cpu, gpu, fpga_fast_emu };

enum Errors : int8_t {
  OPENCL_AOT_FILE_NOT_EXIST = 1,
  OPENCL_AOT_FAILED_TO_CREATE_OPENCL_PROGRAM,
  OPENCL_AOT_FAILED_TO_CREATE_ELF,
  OPENCL_AOT_DEVICE_INFO_PARAMETER_IS_EMPTY,
  OPENCL_AOT_LIST_OF_INPUT_FILES_IS_EMPTY,
  OPENCL_AOT_FAILED_TO_OPEN_FILE,
  OPENCL_AOT_FILE_IS_EMPTY,
  OPENCL_AOT_DEVICE_DOESNT_SUPPORT_SPIRV,
  OPENCL_AOT_INPUT_BINARY_FORMAT_IS_NOT_SUPPORTED,
  OPENCL_AOT_OPTIONS_COEXISTENCE_FAILURE,
  OPENCL_AOT_TARGET_CPU_ARCH_FAILURE,
  OPENCL_AOT_DEVICE_ID_IS_EMPTY,
  OPENCL_AOT_PROGRAM_IS_EMPTY,
  OPENCL_AOT_PLATFORM_NOT_FOUND
};

inline bool clFailed(cl_int ReturnCode) { return CL_SUCCESS != ReturnCode; }

std::string getOpenCLErrorNameByErrorCode(cl_int Err);

inline std::string formatCLError(const std::string &Message, cl_int CLErr) {
  return Message + ": " + std::to_string(CLErr) + " (" +
         getOpenCLErrorNameByErrorCode(CLErr) + ")\n";
}

/*! \brief Get OpenCL platform (Intel only) by OpenCL device type
 * \param Type (DeviceType).
 * \return a tuple of OpenCL platform ID, OpenCL platform name, error message
 * and return code.
 */
std::tuple<cl_platform_id, std::string, std::string, cl_int>
getOpenCLPlatform(DeviceType Type);

/*! \brief Get OpenCL device (Intel only) for the OpenCL platform ID
 * \param PlatformId (cl_platform_id).
 * \param Type (DeviceType).
 * \return a tuple of OpenCL platform ID, OpenCL platform name, error message
 * and return code.
 */
std::tuple<cl_device_id, std::string, cl_int>
getOpenCLDevice(cl_platform_id PlatformId, DeviceType Type);

/*! \brief Get information about an OpenCL device.
 * \param DeviceID (cl_device_id).
 * \param ParamName (cl_device_info).
 * \return a tuple of parameter value, error message and return code.
 */
std::tuple<std::string, std::string, cl_int>
getOpenCLDeviceInfo(cl_device_id &DeviceId, cl_device_info ParamName);

/*! \brief Create OpenCL program from IL (SPIR-V, SPIR) and return it.
 * \param IL (std::vector<char>).
 * \param Context (cl_context).
 * \param PlatformId (cl_platform_id).
 * \param DeviceId (cl_device_id).
 * \return a tuple of OpenCL program, error message and return code.
 */
std::tuple<cl_program, std::string, cl_int>
createProgramWithIL(std::vector<char> IL, cl_context Context,
                    cl_platform_id PlatformId, cl_device_id DeviceId);

/*! \brief Read binary file and return the vector of characters
 * \param FileName (std::string).
 * \return a tuple of vector of characters (content of the file), error message
 * and return code.
 */
std::tuple<std::vector<char>, std::string, cl_int>
readBinaryFile(std::string FileName);

bool isFileOCLSource(const std::string &FileName);

bool isFileELF(const std::vector<char> &BinaryData);

bool isFileSPIRV(const std::vector<char> &BinaryData);

bool isFileLLVMIR(const std::vector<char> &BinaryData);

#endif /* AOT_COMP_TOOL_UTILS */
