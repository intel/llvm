//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel_compiler_sycl.hpp"
#include <sycl/detail/core.hpp> // get_kernel_ids
#include <sycl/exception.hpp>   // make_error_code

#include <dlfcn.h> // dlopen

#if __GNUC__ && __GNUC__ < 8

// std::filesystem is not availalbe for GCC < 8
// and much of the  cross-platform file handling code depends upon it.
// Given that this extension is experimental and that the file
// handling aspects are most likely temporary, it makes sense to
// simply not support GCC<8.

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

bool SYCL_Compilation_Available() { return false; }

std::vector<sycl::kernel_id>
SYCL_to_Kernel_IDs(const std::string &SYCLSource, include_pairs_t IncludePairs,
                   const std::vector<std::string> &UserArgs,
                   std::string *LogPtr,
                   const std::vector<std::string> &RegisteredKernelNames) {
  (void)SYCLSource;
  (void)IncludePairs;
  (void)UserArgs;
  (void)LogPtr;
  (void)RegisteredKernelNames;
  throw sycl::exception(sycl::errc::build,
                        "kernel_compiler does not support GCC<8");
}
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#else

#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

std::string generateSemiUniqueId() {
  // Get the current time as a time_t object.
  std::time_t CurrentTime = std::time(nullptr);

  // Convert time_t to a string with format YYYYMMDD_HHMMSS.
  std::tm *LocalTime = std::localtime(&CurrentTime);
  std::stringstream Ss;
  Ss << std::put_time(LocalTime, "%Y%m%d_%H%M%S");

  // Amend with random number.
  std::random_device Rd;
  int RandomNumber = Rd() % 900 + 100;
  Ss << "_" << std::setfill('0') << std::setw(3) << RandomNumber;

  return Ss.str();
}

std::filesystem::path prepareWS(const std::string &Id) {
  const std::filesystem::path TmpDirectoryPath =
      std::filesystem::current_path();
  // std::filesystem::temp_directory_path();
  std::filesystem::path NewDirectoryPath = TmpDirectoryPath / Id;

  try {
    std::filesystem::create_directories(NewDirectoryPath);
  } catch (const std::filesystem::filesystem_error &E) {
    throw sycl::exception(sycl::errc::build, E.what());
  }

  return NewDirectoryPath;
}

std::string userArgsAsString(const std::vector<std::string> &UserArguments) {
  return std::accumulate(UserArguments.begin(), UserArguments.end(),
                         std::string(""),
                         [](const std::string &A, const std::string &B) {
                           return A.empty() ? B : A + " " + B;
                         });
}

void outputPreamble(std::ofstream &Os, const std::filesystem::path &FilePath,
                    const std::string &Id,
                    const std::vector<std::string> &UserArgs) {

  Os << "/*\n";
  Os << "  clang++ -fsycl -fPIC -shared -o " << Id << ".so ";
  Os << userArgsAsString(UserArgs);
  Os << Id << ".cpp \n */" << std::endl;
}

std::filesystem::path
outputCpp(const std::filesystem::path &ParentDir, const std::string &Id,
          std::string RawCodeString, const std::vector<std::string> &UserArgs,
          const std::vector<std::string> &RegisteredKernelNames) {
  std::filesystem::path FilePath = ParentDir / (Id + ".cpp");
  std::ofstream Outfile(FilePath, std::ios::out | std::ios::trunc);

  if (Outfile.is_open()) {
    outputPreamble(Outfile, FilePath, Id, UserArgs);
    Outfile << RawCodeString << std::endl;

    Outfile << "int temp() {\n";
    for (const std::string &KernelName : RegisteredKernelNames) {
      Outfile << "  " << KernelName << ";\n";
    }
    Outfile << "  return 0;\n}\n" << std::endl;

    Outfile.close();
  } else {
    throw sycl::exception(sycl::errc::build,
                          "Failed to open .cpp file for write: " +
                              FilePath.string());
  }

  return FilePath;
}

void outputIncludeFiles(const std::filesystem::path &Dirpath,
                        include_pairs_t IncludePairs) {
  using pairStrings = std::pair<std::string, std::string>;
  for (pairStrings p : IncludePairs) {
    std::filesystem::path FilePath = Dirpath / p.first;
    std::filesystem::create_directories(FilePath.parent_path());
    std::ofstream outfile(FilePath, std::ios::out | std::ios::trunc);
    if (outfile.is_open()) {
      outfile << p.second << std::endl;

      outfile.close();
    } else {
      throw sycl::exception(sycl::errc::build,
                            "Failed to open include file for write: " +
                                FilePath.string());
    }
  }
}

std::string getCompilerName() {
#ifdef __WIN32
  std::string Compiler = "clang++.exe";
#else
  std::string Compiler = "clang++";
#endif
  return Compiler;
}

std::filesystem::path invokeCompiler(const std::filesystem::path &FPath,
                                     const std::filesystem::path &DPath,
                                     const std::string &Id,
                                     const std::vector<std::string> &UserArgs,
                                     std::string *LogPtr) {

#ifdef _WIN32
  std::string fPIC = "";
  std::string Ext = ".dll";
#else
  std::string fPIC = "-fPIC ";
  std::string Ext = ".so";
#endif

  std::filesystem::path FilePath(FPath);
  std::filesystem::path ParentDir(DPath);
  std::filesystem::path TargetPath = ParentDir / (Id + Ext);
  std::filesystem::path LogPath = ParentDir / "compilation_log.txt";
  std::string Compiler = getCompilerName();

  std::string Command = Compiler + " -fsycl -shared " + fPIC + " -o " +
                        TargetPath.make_preferred().string() + " " +
                        userArgsAsString(UserArgs) + " " +
                        FilePath.make_preferred().string() + " 2> " +
                        LogPath.make_preferred().string();

  int Result = std::system(Command.c_str());

  // Read the log file contents into the log variable.
  std::string CompileLog;
  std::ifstream LogStream;
  LogStream.open(LogPath);
  if (LogStream.is_open()) {
    std::stringstream LogBuffer;
    LogBuffer << LogStream.rdbuf();
    CompileLog.append(LogBuffer.str());
    if (LogPtr != nullptr)
      LogPtr->append(LogBuffer.str());

  } else if (Result == 0 && LogPtr != nullptr) {
    // If there was a compilation problem, we want to report that (below),
    // not a mere "missing log" error.
    throw sycl::exception(sycl::errc::build,
                          "failure retrieving compilation log");
  }

  if (Result != 0) {
    throw sycl::exception(sycl::errc::build,
                          "Compile failure: " + std::to_string(Result) + " " +
                              CompileLog);
  }

  return TargetPath;
}

std::vector<sycl::kernel_id>
openSharedLib(const std::filesystem::path &TargetPath) {
  std::vector<sycl::kernel_id> before_kids = sycl::get_kernel_ids();
#ifdef _WIN32
  LoadLibrary(TargetPath.c_str());
#else
  dlopen(TargetPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  std::vector<sycl::kernel_id> after_kids = sycl::get_kernel_ids();
  // std::cout << "all kernels " << std::endl;
  // for(auto &kid: after_kids){
  //   std::cout << " kernel =>  " << kid.get_name() << std::endl;
  // }

  // the new kernels seem to appear first.  They don't support a comparsion
  // operation, so are unsortable. If not this, we would have to brute force
  // diff the before and after vectors, which could potentially be expensive.
  int n = after_kids.size() - before_kids.size();
  std::vector<sycl::kernel_id> difference(after_kids.begin(),
                                          after_kids.begin() + n);

  std::cout << "new kernels " << std::endl;
  for (auto &kid : difference) {
    std::cout << " kernel =>  " << kid.get_name() << std::endl;
  }

  return difference;
}

std::vector<sycl::kernel_id>
SYCL_to_Kernel_IDs(const std::string &SYCLSource, include_pairs_t IncludePairs,
                   const std::vector<std::string> &UserArgs,
                   std::string *LogPtr,
                   const std::vector<std::string> &RegisteredKernelNames) {
  // clang-format off
  const std::string id                    = generateSemiUniqueId();
  const std::filesystem::path ParentDir   = prepareWS(id);
  std::filesystem::path FilePath          = outputCpp(ParentDir, id, SYCLSource, UserArgs, RegisteredKernelNames);
                                            outputIncludeFiles(ParentDir, IncludePairs);
  const std::filesystem::path TargetPath  = invokeCompiler(FilePath, ParentDir, id, UserArgs, LogPtr);
                                    return  openSharedLib(TargetPath);
  // clang-format on
}

bool SYCL_Compilation_Available() {
  // Is compiler on $PATH ? We try to invoke it.
  std::string id = generateSemiUniqueId();
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  std::filesystem::path DumpPath = tmp / (id + "_version.txt");
  std::string Compiler = getCompilerName();
  std::string TestCommand =
      Compiler + " --version &> " + DumpPath.make_preferred().string();
  int result = std::system(TestCommand.c_str());

  return (result == 0);
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
#endif
