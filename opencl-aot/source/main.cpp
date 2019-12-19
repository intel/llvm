//===------ opencl-aot/source/main.cpp - opencl-aot tool --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the opencl-aot tool which generates
/// OpenCL program binary from SPIR-V binary for Intel(R) processor, Intel(R)
/// Processor Graphics and Intel(R) FPGA Emulation Platform for OpenCL(TM)
/// devices.
///
//===----------------------------------------------------------------------===//

#include "utils.h"
#include <CL/cl.h>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

// cl_program, shall initialized with
auto ProgramDeleter = [](cl_program CLProg) {
  if (CLProg != nullptr) {
    cl_int CLErr = clReleaseProgram(CLProg);
    if (clFailed(CLErr)) {
      std::cerr << formatCLError("Failed to release OpenCL program", CLErr);
    }
  }
};

using CLProgramUPtr = std::unique_ptr<
    _cl_program, decltype(ProgramDeleter)>; // decltype(CLProgramUPtr.get()) is
                                            // same as cl_program, should
                                            // initialized with ProgramDeleter

auto ContextDeleter = [](cl_context CLCon) {
  if (CLCon != nullptr) {
    cl_int CLErr = clReleaseContext(CLCon);
    if (clFailed(CLErr)) {
      std::cerr << formatCLError("Failed to release OpenCL context", CLErr);
    }
  }
};

using CLContextUPtr = std::unique_ptr<
    _cl_context, decltype(ContextDeleter)>; // decltype(CLContextUPtr.get()) is
                                            // same as cl_context, should
                                            // initialized with ContextDeleter

/*! \brief Generate OpenCL program from OpenCL program binary (in ELF format) or
 * SPIR-V binary file
 * \param FileNames (const std::vector<std::string>).
 * \param PlatformId (cl_platform_id).
 * \param DeviceId (cl_device_id).
 * \return a tuple of vector of unique pointers to programs, error message and
 * return code.
 */
std::tuple<std::vector<CLProgramUPtr>, std::string, cl_int>
generateProgramsFromBinaries(const std::vector<std::string> &FileNames,
                             cl_platform_id PlatformId, cl_device_id DeviceId) {
  // step 0: define internal types

  enum SupportedTypes : int8_t { BEGIN, ELF = BEGIN, SPIRV, UNKNOWN, END };
  static std::map<SupportedTypes, const std::string> SupportedTypesToNames{
      {ELF, "OpenCL program binary"}, {SPIRV, "SPIR-V"}, {UNKNOWN, "UNKNOWN"}};
  assert(SupportedTypesToNames.size() == SupportedTypes::END &&
         "unexpected name"); // to prevent incorrect refactoring

  enum { FTYPE, DATA };

  using Content = std::pair<SupportedTypes, std::vector<char>>;

  // step 1: validate input

  if (FileNames.empty() ||
      std::any_of(FileNames.begin(), FileNames.end(),
                  [](const std::string &File) { return File.empty(); })) {
    return std::make_tuple(
        std::vector<CLProgramUPtr>(),
        "List of input binaries is empty or contains empty filename\n",
        OPENCL_AOT_LIST_OF_INPUT_BINARIES_IS_EMPTY);
  }

  assert(PlatformId != nullptr && "Platform must not be empty");
  assert(DeviceId != nullptr && "Device must not be empty");

  // step 2: read files and put them to cache

  std::unordered_map<std::string, Content> FileNameToContentMap;
  enum { FNAME, CONTENT };
  FileNameToContentMap.reserve(FileNames.size());

  cl_int CLErr(CL_SUCCESS);
  std::string ErrorMessage;
  for (const std::string &FileName : FileNames) {
    std::vector<char> FileContent;
    std::tie(FileContent, ErrorMessage, CLErr) = readBinaryFile(FileName);

    if (clFailed(CLErr)) {
      return std::make_tuple(std::vector<CLProgramUPtr>(), ErrorMessage, CLErr);
    }

    FileNameToContentMap.emplace(
        FileName,
        std::make_pair(SupportedTypes::UNKNOWN, std::move(FileContent)));
  }

  // step 3: find out type of each file

  for (auto &Value : FileNameToContentMap) {
    Content &FileContent = std::get<CONTENT>(Value);
    auto &FileType = std::get<FTYPE>(FileContent);
    const auto &BinaryData = std::get<DATA>(FileContent);
    const auto &FileName = std::get<FNAME>(Value);

    if (isFileELF(BinaryData)) {
      FileType = ELF;
    } else if (isFileSPIRV(BinaryData)) {
      FileType = SPIRV;
    } else {
      FileType = UNKNOWN;
      return std::make_tuple(
          std::vector<CLProgramUPtr>(),
          "Unable to identify the format of input binary file " + FileName +
              '\n',
          OPENCL_AOT_INPUT_BINARY_FORMAT_IS_NOT_SUPPORTED);
    }
  }

  // step 4: create OpenCL program from binary file (unique for each type)

  std::vector<cl_program> Programs;
  Programs.reserve(FileNames.size());

  cl_context Context =
      clCreateContext(nullptr, 1, &DeviceId, nullptr, nullptr, &CLErr);
  CLContextUPtr ContextUPtr(Context, ContextDeleter);
  if (clFailed(CLErr)) {
    return std::make_tuple(std::vector<CLProgramUPtr>(),
                           formatCLError("Failed to create context", CLErr),
                           CLErr);
  }

  for (const auto &Value : FileNameToContentMap) {
    const Content &FileContent = std::get<CONTENT>(Value);
    const auto &FileType = std::get<FTYPE>(FileContent);
    const auto &BinaryData = std::get<DATA>(FileContent);
    const auto &FileName = std::get<FNAME>(Value);

    cl_program Program(nullptr);

    cl_int BinaryStatus(CL_SUCCESS);

    switch (FileType) {
    case ELF: {
      const size_t BinarySize = BinaryData.size();
      const auto *BinaryDataRaw =
          reinterpret_cast<const unsigned char *>(BinaryData.data());
      Program = clCreateProgramWithBinary(
          ContextUPtr.get(), /* Number of devices = */ 1, &DeviceId,
          &BinarySize, &BinaryDataRaw, &BinaryStatus, &CLErr);
      break;
    }
    case SPIRV: {
      std::tie(Program, ErrorMessage, CLErr) = createProgramWithIL(
          BinaryData, ContextUPtr.get(), PlatformId, DeviceId);
      if (clFailed(CLErr)) {
        return std::make_tuple(std::vector<CLProgramUPtr>(), ErrorMessage,
                               CLErr);
      }
      break;
    }
    default:
      assert(FileType != UNKNOWN &&
             "Unable to identify the format of input binary file");
    }

    if (clFailed(CLErr) || clFailed(BinaryStatus) || Program == nullptr) {
      ErrorMessage = "Failed to create OpenCL program from " +
                     SupportedTypesToNames[FileType] + " file " + FileName +
                     '\n';
      ErrorMessage += "Binary status: " + std::to_string(BinaryStatus) + '\n';
      ErrorMessage += formatCLError("Error code: ", CLErr) + '\n';
      return std::make_tuple(std::vector<CLProgramUPtr>(), ErrorMessage,
                             OPENCL_AOT_FAILED_TO_CREATE_OPENCL_PROGRAM);
    }

    Programs.push_back(Program);

    std::cout << "OpenCL program was successfully created from " +
                     SupportedTypesToNames[FileType] + " file " + FileName
              << '\n';
  }

  // step 5: create guards to return result safely

  std::vector<CLProgramUPtr> Result;

  for (const auto &Program : Programs) {
    Result.emplace_back(CLProgramUPtr(Program, ProgramDeleter));
  }

  if (Result.empty()) {
    ErrorMessage = "Failed to create OpenCL program";
    CLErr = OPENCL_AOT_FAILED_TO_CREATE_OPENCL_PROGRAM;
  }
  return std::make_tuple(std::move(Result), ErrorMessage, CLErr);
}

std::tuple<std::string, std::string, cl_int>
getCompilerBuildLog(const CLProgramUPtr &ProgramUPtr, cl_device_id Device) {
  cl_int CLErr(CL_SUCCESS);
  size_t BuildLogLength = 0;
  CLErr = clGetProgramBuildInfo(ProgramUPtr.get(), Device, CL_PROGRAM_BUILD_LOG,
                                0, nullptr, &BuildLogLength);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        "", formatCLError("Failed to get size of program build log", CLErr),
        CLErr);
  }

  std::string BuildLog(
      BuildLogLength - 1,
      '\0'); // std::string object adds another \0 to the end of the string, so
             // subtract 1 to avoid extra space in the of the string

  CLErr = clGetProgramBuildInfo(ProgramUPtr.get(), Device, CL_PROGRAM_BUILD_LOG,
                                BuildLogLength, &BuildLog.front(), nullptr);
  if (clFailed(CLErr)) {
    return std::make_tuple(
        "", formatCLError("Failed to get program build log", CLErr), CLErr);
  }

  // don't check if compiler build log is empty and don't set error message
  // because it could be a correct case
  return std::make_tuple(std::string(BuildLog.begin(), BuildLog.end()), "",
                         CLErr);
}

int main(int Argc, char *Argv[]) {
  // step 0: set command line options
  cl::opt<std::string> OptInputBinary(
      cl::Positional, cl::Required,
      cl::desc("<input SPIR-V or OpenCL program binary>"),
      cl::value_desc("filename"));
  cl::opt<DeviceType> OptDevice(
      "device", cl::Required, cl::desc("Set target device type:"),
      cl::values(
          clEnumVal(cpu, "Intel(R) processor device"),
          clEnumVal(gpu, "Intel(R) Processor Graphics device"),
          clEnumVal(fpga_fast_emu,
                    "Intel(R) FPGA Emulation Platform for OpenCL(TM) device")));
  enum ArchType : int8_t { sse42 = 0, avx, avx2, avx512 };
  cl::opt<ArchType> OptMArch(
      "march",
      cl::desc("Set target CPU architecture (for --device=cpu or "
               "--device=fpga_fast_emu only):"),
      cl::values(
          clEnumVal(
              avx512,
              "Enable support of Intel(R) Advanced Vector Extensions 512 "
              "(Intel(R) AVX-512) Foundation Instructions, Intel(R) AVX-512 "
              "Conflict Detection Instructions, Intel(R) AVX-512 Doubleword "
              "and Quadword Instructions, Intel(R) AVX-512 Byte and Word "
              "Instructions and Intel(R) AVX-512 Vector Length Extensions for "
              "Intel(R) processors and the instructions enabled with "
              "-march=avx2"),
          clEnumVal(avx2,
                    "Enable support of Intel(R) Advanced Vector Extensions 2 "
                    "(Intel(R) AVX2), Intel(R) Advanced Vector Extensions "
                    "(Intel(R) AVX), Intel(R) Streaming SIMD Extensions 4.2 "
                    "(Intel(R) SSE4.2), Intel(R) SSE4.1, Intel(R) SSE3, "
                    "Intel(R) SSE2, Intel(R) SSE, and Supplemental Streaming "
                    "SIMD Extensions 3 (SSSE3) instructions"),
          clEnumVal(avx, "Enable support of Intel(R) AVX, Intel(R) SSE4.2, "
                         "Intel(R) SSE4.1, Intel(R) SSE3, Intel(R) SSE2, "
                         "Intel(R) SSE, and SSSE3 instructions"),
          clEnumValN(
              sse42, "sse4.2",
              "Enable support of Intel(R) SSE4.2 Efficient Accelerated String "
              "and Text Processing Instructions, Intel(R) SSE4 Vectorizing "
              "Compiler and Media Accelerator, Intel(R) SSE3, Intel(R) SSE2, "
              "Intel(R) SSE, and SSSE3 instructions")));
  cl::opt<std::string> OptOutputElf(
      "o", cl::init("output.bin"),
      cl::desc("Specify the output OpenCL program binary filename"),
      cl::value_desc("filename"));
  cl::opt<std::string> OptBuildOptions("bo",
                                       cl::desc("Set OpenCL build options"),
                                       cl::value_desc("build options"));

  cl::ParseCommandLineOptions(Argc, Argv,
                              "OpenCL ahead-of-time (AOT) compilation tool");

  // step 1: perform checks for command line options
  bool IsFileExists = false;
  sys::fs::is_regular_file(OptInputBinary, IsFileExists);
  if (!IsFileExists) {
    std::cerr << "File " << OptInputBinary << " does not exist!" << '\n';
    return OPENCL_AOT_FILE_NOT_EXIST;
  }

  if (OptMArch.getNumOccurrences() &&
      (OptDevice.getValue() != cpu && OptDevice.getValue() != fpga_fast_emu)) {
    std::cerr << "Use --march option with --device=cpu or "
                 "--device=fpga_fast_emu only";
    return OPENCL_AOT_OPTIONS_COEXISTENCE_FAILURE;
  }

  cl_int CLErr(CL_SUCCESS);

  // step 2: get OpenCL platform
  cl_platform_id PlatformId = nullptr;
  std::string PlatformName;
  std::string ErrorMessage;
  std::tie(PlatformId, PlatformName, ErrorMessage, CLErr) =
      getOpenCLPlatform(OptDevice);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  std::cout << "Platform name: " << PlatformName << '\n';

  // step 3: get OpenCL device
  cl_device_id DeviceId = nullptr;
  std::tie(DeviceId, ErrorMessage, CLErr) =
      getOpenCLDevice(PlatformId, OptDevice);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  std::string DeviceName;
  std::tie(DeviceName, ErrorMessage, CLErr) =
      getOpenCLDeviceInfo(DeviceId, CL_DEVICE_NAME);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  std::cout << "Device name: " << DeviceName << '\n';

  // step 4: get driver version
  std::string DriverVersion;
  std::tie(DriverVersion, ErrorMessage, CLErr) =
      getOpenCLDeviceInfo(DeviceId, CL_DRIVER_VERSION);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  std::cout << "Driver version: " << DriverVersion << '\n';

  // step 5: enable optimizations for target CPU architecture
  if (OptMArch.getNumOccurrences()) {
    std::string CPUTargetArchEnvVarName = "CL_CONFIG_CPU_TARGET_ARCH";
    std::map<ArchType, std::string> ArchTypeToCPUTargetArchEnvVarValues{
        {sse42, "corei7"},
        {avx, "corei7-avx"},
        {avx2, "core-avx2"},
        {avx512, "skx"}};
    int EnvErr = 0;
#ifdef _WIN32
    EnvErr = _putenv(std::string(CPUTargetArchEnvVarName + "=" +
                                 ArchTypeToCPUTargetArchEnvVarValues[OptMArch])
                         .c_str());
#else
    EnvErr = setenv(CPUTargetArchEnvVarName.c_str(),
                    ArchTypeToCPUTargetArchEnvVarValues[OptMArch].c_str(), 1);
#endif
    std::map<ArchType, std::string> ArchTypeToArchTypeName{
        {sse42, "Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2)"},
        {avx, "Intel(R) Advanced Vector Extensions (Intel(R) AVX)"},
        {avx2, "Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)"},
        {avx512, "Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512)"}};
    if (EnvErr) {
      std::cerr << "Failed to set target CPU architecture to "
                << ArchTypeToArchTypeName[OptMArch] << '\n';
      return OPENCL_AOT_TARGET_CPU_ARCH_FAILURE;
    }
    std::cout << "Setting target CPU architecture to "
              << ArchTypeToArchTypeName[OptMArch] << '\n';
  }

  // step 6: generate OpenCL programs from input binaries
  std::vector<CLProgramUPtr> Progs;
  std::tie(Progs, ErrorMessage, CLErr) = generateProgramsFromBinaries(
      {OptInputBinary.c_str()}, PlatformId, DeviceId);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  CLProgramUPtr ProgramUPtr(std::move(Progs[0]));

  // step 7: set OpenCL build options
  std::string BuildOptions = OptBuildOptions;
  auto ParentDir = sys::path::parent_path(OptInputBinary);
  if (!ParentDir.empty()) {
    BuildOptions += " -I \"" + std::string(ParentDir) + '\"';
  }
  std::cout << "Using build options: " << BuildOptions << '\n';

  // step 8: build OpenCL program
  CLErr = clBuildProgram(ProgramUPtr.get(), 1, &DeviceId,
                         OptBuildOptions.c_str(), nullptr, nullptr);

  std::string CompilerBuildLog;
  std::tie(CompilerBuildLog, ErrorMessage, std::ignore) =
      getCompilerBuildLog(ProgramUPtr, DeviceId);

  if (!ErrorMessage.empty()) {
    std::cerr << ErrorMessage;
    // don't exit because we should show compiler build log and/or error message
    // from clBuildProgram if it will be
  }

  if (!CompilerBuildLog.empty()) {
    std::cout << CompilerBuildLog << '\n';
  }

  if (clFailed(CLErr)) {
    std::cerr << formatCLError("Failed to build a program:", CLErr) << '\n';
    return CLErr;
  }

  // step 9: get program binary
  size_t ProgramBinarySize = 0;
  CLErr = clGetProgramInfo(ProgramUPtr.get(), CL_PROGRAM_BINARY_SIZES,
                           sizeof(size_t), &ProgramBinarySize, nullptr);
  if (clFailed(CLErr) || ProgramBinarySize == 0) {
    std::cerr << formatCLError("Failed to get OpenCL program binary size",
                               CLErr)
              << '\n';
    return CLErr;
  }

  std::vector<unsigned char> ProgramBinaries(ProgramBinarySize, '\0');
  auto ProgramBinariesRaw = ProgramBinaries.data();
  CLErr =
      clGetProgramInfo(ProgramUPtr.get(), CL_PROGRAM_BINARIES,
                       sizeof(unsigned char *), &ProgramBinariesRaw, nullptr);
  if (clFailed(CLErr)) {
    std::cerr << formatCLError("Failed to get OpenCL program binary data",
                               CLErr)
              << '\n';
    return CLErr;
  }

  // step 10: write program binary (in ELF format) to the file
  std::ofstream OutputELF(OptOutputElf, std::ofstream::binary);

  for (const auto &Chunk : ProgramBinaries) {
    OutputELF << Chunk;
  }

  if (!OutputELF.good()) {
    std::cerr << "Failed to create OpenCL program binary file" << '\n';
    return OPENCL_AOT_FAILED_TO_CREATE_ELF;
  }
  std::cout << "OpenCL program binary file was successfully created: "
            << OptOutputElf << '\n';

  return CL_SUCCESS;
}
