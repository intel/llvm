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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
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

using CLContextSPtr =
    std::shared_ptr<_cl_context>; // decltype(CLContextSPtr.get()) is
                                  // same as cl_context, should
                                  // initialized with ContextDeleter

static bool verbose = false;
static llvm::raw_ostream &logs() {
  static llvm::raw_ostream &logger = verbose ? llvm::outs() : llvm::nulls();
  return logger;
}

/*! \brief Generate OpenCL program from OpenCL program binary (in ELF format) or
 * SPIR-V binary file or LLVM IR (bitcode file) or OpenCL source file
 * \param FileNames (const std::vector<std::string>).
 * \param PlatformId (cl_platform_id).
 * \param DeviceId (cl_device_id).
 * \param ContextSPtr (CLContextSPtr).
 * \return a tuple of vector of unique pointers to programs, error message and
 * return code.
 */
std::tuple<std::vector<CLProgramUPtr>, std::string, cl_int>
generateProgramsFromInput(const std::vector<std::string> &FileNames,
                          cl_platform_id PlatformId, cl_device_id DeviceId,
                          CLContextSPtr ContextSPtr) {
  // step 0: define internal types

  enum SupportedTypes : int8_t {
    BEGIN,
    ELF = BEGIN,
    SPIRV,
    LLVMIR,
    SOURCE,
    UNKNOWN,
    END
  };
  static std::map<SupportedTypes, const std::string> SupportedTypesToNames{
      {ELF, "OpenCL program binary"},
      {SPIRV, "SPIR-V"},
      {LLVMIR, "LLVM IR bitcode"},
      {SOURCE, "OpenCL source"},
      {UNKNOWN, "UNKNOWN"}};
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
        OPENCL_AOT_LIST_OF_INPUT_FILES_IS_EMPTY);
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

    if (isFileOCLSource(FileName)) {
      FileType = SOURCE;
    } else if (isFileELF(BinaryData)) {
      FileType = ELF;
    } else if (isFileSPIRV(BinaryData)) {
      FileType = SPIRV;
    } else if (isFileLLVMIR(BinaryData)) {
      FileType = LLVMIR;
    } else {
      FileType = UNKNOWN;
      return std::make_tuple(
          std::vector<CLProgramUPtr>(),
          "Unable to identify the format of input binary file " + FileName +
              '\n',
          OPENCL_AOT_INPUT_BINARY_FORMAT_IS_NOT_SUPPORTED);
    }
  }

  // step 4: create OpenCL program from input files (unique for each type)

  std::vector<cl_program> Programs;
  Programs.reserve(FileNames.size());

  for (const auto &Value : FileNameToContentMap) {
    const Content &FileContent = std::get<CONTENT>(Value);
    const auto &FileType = std::get<FTYPE>(FileContent);
    const auto &BinaryData = std::get<DATA>(FileContent);
    const auto &FileName = std::get<FNAME>(Value);

    cl_program Program(nullptr);

    cl_int BinaryStatus(CL_SUCCESS);

    switch (FileType) {
    case LLVMIR:
    case ELF: {
      const size_t BinarySize = BinaryData.size();
      const auto *BinaryDataRaw =
          reinterpret_cast<const unsigned char *>(BinaryData.data());
      Program = clCreateProgramWithBinary(
          ContextSPtr.get(), /* Number of devices = */ 1, &DeviceId,
          &BinarySize, &BinaryDataRaw, &BinaryStatus, &CLErr);
      break;
    }
    case SPIRV: {
      std::tie(Program, ErrorMessage, CLErr) = createProgramWithIL(
          BinaryData, ContextSPtr.get(), PlatformId, DeviceId);
      if (clFailed(CLErr)) {
        return std::make_tuple(std::vector<CLProgramUPtr>(), ErrorMessage,
                               CLErr);
      }
      break;
    }
    case SOURCE: {
      const size_t SourceSize = BinaryData.size();
      const auto *Source = reinterpret_cast<const char *>(BinaryData.data());
      Program = clCreateProgramWithSource(ContextSPtr.get(), 1, &Source,
                                          &SourceSize, &CLErr);
      break;
    }
    default:
      assert(FileType != UNKNOWN &&
             "Unable to identify the format of input file");
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

    logs() << "OpenCL program was successfully created from " +
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
  cl::list<std::string> OptInputFileList(
      cl::Positional, cl::ZeroOrMore,
      cl::desc(
          "<input file(s)>\n\n"
          "Supported type of input file:\n"
          "* SOURCE: OpenCL C source (.cl)\n"
          "* BINARY: SPIR-V, LLVM IR bitcode, OpenCL compiled object(ELF)"),
      cl::value_desc("filename"));
  cl::opt<std::string> OptOutputElf(
      "o", cl::init("aot-output"),
      cl::desc("Specify the output OpenCL program object or binary filename"),
      cl::value_desc("filename"));

  cl::opt<DeviceType> OptDevice(
      "device", cl::Required, cl::desc("Set target device type:"),
      cl::values(
          clEnumVal(cpu, "Intel(R) processor device"),
          clEnumVal(gpu, "Intel(R) Processor Graphics device"),
          clEnumVal(fpga_fast_emu,
                    "Intel(R) FPGA Emulation Platform for OpenCL(TM) device")));

  enum Commands : int8_t { build, compile, link };
  cl::opt<Commands> OptCommand(
      "cmd", cl::desc("Command"), cl::init(build),
      cl::values(
          clEnumVal(build,
                    "Build (compile and link) OpenCL program binary from SPIRV "
                    "or OpenCL program object or OpenCL C source"),
          clEnumVal(compile,
                    "Compile OpenCL C source to OpenCL program object"),
          clEnumVal(link, "Link OpenCL compiled program objects and creates "
                          "OpenCL program binary")));

  // TODO: These options are only needed for compatibility with old tool.
  // Delete these options once transition is done.
  cl::alias OptInput("input", cl::aliasopt(OptInputFileList),
                     cl::desc("Input OpenCL C source file"));
  cl::alias OptSPIRV("spv", cl::aliasopt(OptInputFileList),
                     cl::desc("Input SPIR-V file"));
  cl::alias OptBinaries("binary", cl::aliasopt(OptInputFileList),
                        cl::desc("Input OpenCL program binary files"));
  cl::alias OptIR("ir", cl::aliasopt(OptOutputElf), cl::desc("Output file"));

  enum ArchType : int8_t {
    sse42 = 0,
    avx,
    avx2,
    avx512,
    wsm,
    snb,
    ivyb,
    bdw,
    cfl,
    adl,
    skylake,
    skx,
    clk,
    icl,
    icx,
    spr,
    gnr
  };
  cl::opt<ArchType> OptMArch(
      "march",
      cl::desc("Set target CPU architecture according to specified instruction "
               "set or CPU device "
               "name (for --device=cpu or --device=fpga_fast_emu only):"),
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
              "Intel(R) SSE, and SSSE3 instructions"),
          clEnumVal(wsm, "Intel® microarchitecture code name Westmere"),
          clEnumVal(snb, "Intel® microarchitecture code name Sandy Bridge"),
          clEnumVal(ivyb, "Intel® microarchitecture code name Ivy Bridge"),
          clEnumVal(bdw, "Intel® microarchitecture code name Broadwell"),
          clEnumVal(cfl, "Coffee Lake"), clEnumVal(adl, "Alder Lake"),
          clEnumVal(skylake,
                    "Intel® microarchitecture code name Skylake (client)"),
          clEnumVal(skx, "Intel® microarchitecture code name Skylake (server)"),
          clEnumVal(clk, "Cascade Lake"), clEnumVal(icl, "Ice Lake (client)"),
          clEnumVal(icx, "Ice Lake (server)"),
          clEnumVal(spr, "Sapphire Rapids"), clEnumVal(gnr, "Granite Rapids")));
  cl::list<std::string> OptBuildOptions("bo", cl::ZeroOrMore,
                                        cl::desc("Set OpenCL build options"),
                                        cl::value_desc("build options"));
  cl::opt<bool, true> OptVerbose("verbose", cl::desc("Show verbose logs"),
                                 cl::location(verbose));
  cl::alias OptV("v", cl::aliasopt(OptVerbose));

  cl::ParseCommandLineOptions(
      Argc, Argv,
      "OpenCL ahead-of-time (AOT) compilation tool\n\n"
      "This program is a OpenCL kernel build tool, which accepts OpenCL C "
      "source, LLVM IR bitcode, OpenCL compiled object or SPIR-V as input. "
      "In addition, separated building (compiling and linking) is optional.\n");

  // step 1: perform checks for command line options

  std::map<Commands, std::pair<std::string, std::string>> CmdToCmdInfoMap = {
      {Commands::build, {"build", "binary"}},
      {Commands::compile, {"compile", "object"}},
      {Commands::link, {"link", "binary"}}};

  std::vector<std::string> InputFileNames;
  for (const auto &FN : OptInputFileList) {
    std::stringstream SS(FN);
    std::string Item;
    // --binary accepts comma-separated binary file list.
    while (std::getline(SS, Item, ',')) {
      bool IsFileExists = false;
      sys::fs::is_regular_file(Item, IsFileExists);
      if (!IsFileExists) {
        std::cerr << "File " << Item << " does not exist!" << '\n';
        return OPENCL_AOT_FILE_NOT_EXIST;
      }

      // If not a link command, we only handle one input file each time.
      if (OptCommand != Commands::link && InputFileNames.size() >= 1) {
        logs() << "WARNING: Can " << CmdToCmdInfoMap[OptCommand].first
               << " only 1 file each time. Extra file(s) will be ignored!\n";
        break;
      }
      InputFileNames.push_back(Item);
    }
  }

  if (InputFileNames.empty()) {
    std::cerr << "no input file.\n";
    return OPENCL_AOT_LIST_OF_INPUT_FILES_IS_EMPTY;
  }

  std::string OutputFileName(OptOutputElf);
  if (!OptOutputElf.getNumOccurrences()) {
    // Copy input file name without extension to output file name if there is
    // only one input file and -o is not specified.
    if (InputFileNames.size() == 1)
      OutputFileName = InputFileNames[0].substr(0, InputFileNames[0].find('.'));
    // Append extension to output file name.
    OutputFileName += OptCommand == Commands::compile
                          ? ".obj"
                          : OptDevice == fpga_fast_emu ? ".aocx" : ".bin";
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

  logs() << "Platform name: " << PlatformName << '\n';

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

  logs() << "Device name: " << DeviceName << '\n';

  // step 4: get driver version
  std::string DriverVersion;
  std::tie(DriverVersion, ErrorMessage, CLErr) =
      getOpenCLDeviceInfo(DeviceId, CL_DRIVER_VERSION);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  logs() << "Driver version: " << DriverVersion << '\n';

  // step 5: enable optimizations for target CPU architecture
  if (OptMArch.getNumOccurrences()) {
    std::string CPUTargetArchEnvVarName = "CL_CONFIG_CPU_TARGET_ARCH";
    std::map<ArchType, std::string> ArchTypeToCPUTargetArchEnvVarValues{
        {sse42, "corei7"},       {avx, "corei7-avx"},
        {avx2, "core-avx2"},     {avx512, "skx"},
        {wsm, "corei7"},         {snb, "corei7-avx"},
        {ivyb, "corei7-avx"},    {bdw, "core-avx2"},
        {cfl, "core-avx2"},      {adl, "core-avx2"},
        {skylake, "core-avx2"},  {skx, "skx"},
        {clk, "cascadelake"},    {icl, "icelake-client"},
        {icx, "icelake-server"}, {spr, "sapphirerapids"},
        {gnr, "graniterapids"}};
    int EnvErr = 0;
#ifdef _WIN32
    EnvErr = _putenv(std::string(CPUTargetArchEnvVarName + "=" +
                                 ArchTypeToCPUTargetArchEnvVarValues[OptMArch])
                         .c_str());
#else
    EnvErr = setenv(CPUTargetArchEnvVarName.c_str(),
                    ArchTypeToCPUTargetArchEnvVarValues[OptMArch].c_str(), 1);
#endif
    if (EnvErr) {
      std::cerr << "Failed to set target CPU architecture to "
                << ArchTypeToCPUTargetArchEnvVarValues[OptMArch] << '\n';
      return OPENCL_AOT_TARGET_CPU_ARCH_FAILURE;
    }
    logs() << "Setting target CPU architecture to "
           << ArchTypeToCPUTargetArchEnvVarValues[OptMArch] << '\n';
  }

  // step 6: generate OpenCL programs from input files

  // Create context
  cl_context Context =
      clCreateContext(nullptr, 1, &DeviceId, nullptr, nullptr, &CLErr);
  CLContextSPtr ContextSPtr(Context, ContextDeleter);
  if (clFailed(CLErr)) {
    std::cerr << formatCLError("Failed to create context", CLErr) << '\n';
  }

  std::vector<CLProgramUPtr> Progs;
  std::tie(Progs, ErrorMessage, CLErr) = generateProgramsFromInput(
      InputFileNames, PlatformId, DeviceId, ContextSPtr);

  if (clFailed(CLErr)) {
    std::cerr << ErrorMessage;
    return CLErr;
  }

  CLProgramUPtr ProgramUPtr(std::move(Progs[0]));

  // step 7: set OpenCL build options
  std::string BuildOptions;
  if (!OptBuildOptions.empty()) {
    for (const auto &BO : OptBuildOptions)
      BuildOptions += BO + ' ';
  }

  // clLinkProgram doesn't accept -I option
  if (OptCommand != Commands::link) {
    assert(!InputFileNames.empty() && "Input file list can't be empty!");
    auto ParentDir = sys::path::parent_path(InputFileNames[0]);
    if (!ParentDir.empty()) {
      BuildOptions += " -I \"" + std::string(ParentDir) + '\"';
    }
    logs() << "Using build options: " << BuildOptions << '\n';
  }

  // step 8: compile | build | link OpenCL program
  switch (OptCommand) {
  case Commands::compile:
    CLErr =
        clCompileProgram(ProgramUPtr.get(), 1, &DeviceId, BuildOptions.c_str(),
                         0, nullptr, nullptr, nullptr, nullptr);
    break;
  case Commands::link: {
    std::vector<cl_program> InputPrograms{ProgramUPtr.get()};
    for (size_t I = 1; I < Progs.size(); ++I)
      InputPrograms.push_back(Progs[I].get());

    cl_program Program = clLinkProgram(
        ContextSPtr.get(), 1, &DeviceId, BuildOptions.c_str(),
        InputPrograms.size(), InputPrograms.data(), nullptr, nullptr, &CLErr);
    ProgramUPtr.reset(Program);
    break;
  }
  default: // Commands::build
    CLErr = clBuildProgram(ProgramUPtr.get(), 1, &DeviceId,
                           BuildOptions.c_str(), nullptr, nullptr);
    break;
  }

  std::string CompilerBuildLog, CompilerBuildLogMessage;
  std::tie(CompilerBuildLog, ErrorMessage, std::ignore) =
      getCompilerBuildLog(ProgramUPtr, DeviceId);

  if (!ErrorMessage.empty()) {
    std::cerr << ErrorMessage;
    // don't exit because we should show compiler build log and/or error message
    // from clBuildProgram if it will be
  }

  if (!CompilerBuildLog.empty()) {
    // According to the return value of getCompilerBuildLog(), ErrorMessage is
    // always empty if CompilerBuildLog is not empty.
    CompilerBuildLogMessage = "\n" + CmdToCmdInfoMap[OptCommand].first +
                              " log:\n" + CompilerBuildLog + '\n';
    logs() << CompilerBuildLogMessage;
  }

  if (clFailed(CLErr)) {
    std::string ErrMsg =
        "Failed to " + CmdToCmdInfoMap[OptCommand].first + ": ";
    // will print CompilerBuildLogMessage when build failed in case verbose is
    // false, in order to provide a friendlier compile error for users.
    if (!verbose)
      std::cerr << CompilerBuildLogMessage;
    std::cerr << formatCLError(ErrMsg, CLErr) << '\n';
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
  std::ofstream OutputELF(OutputFileName, std::ofstream::binary);

  for (const auto &Chunk : ProgramBinaries) {
    OutputELF << Chunk;
  }

  if (!OutputELF.good()) {
    std::cerr << "Failed to create OpenCL program "
              << CmdToCmdInfoMap[OptCommand].second << " file" << '\n';
    return OPENCL_AOT_FAILED_TO_CREATE_ELF;
  }
  logs() << "OpenCL program " << CmdToCmdInfoMap[OptCommand].second
         << " file was successfully created: " << OutputFileName << '\n';

  return CL_SUCCESS;
}
