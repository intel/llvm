//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel_compiler_sycl.hpp"
#include <sycl/exception.hpp> // make_error_code

#include <sycl/detail/os_util.hpp>

#include <ctime>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <stdio.h>

// For GCC versions less than 8, use experimental/filesystem.
#if defined(__has_include) && __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(__has_include) && __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "kernel_compiler sycl requires C++ filesystem support"
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

std::string generateSemiUniqueId() {
  auto Now = std::chrono::high_resolution_clock::now();
  auto Milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      Now.time_since_epoch());

  // Generate random number between 10'000 and 99'900.
  std::random_device RD;
  std::mt19937 Gen(RD());
  std::uniform_int_distribution<int> Distrib(10'000, 99'999);
  int RandomNumber = Distrib(Gen);

  // Combine time and random number into a string.
  std::stringstream Ss;
  Ss << Milliseconds.count() << "_" << RandomNumber;

  return Ss.str();
}

fs::path prepareWS(const std::string &Id) {
  const fs::path TmpDirectoryPath = fs::temp_directory_path();
  fs::path NewDirectoryPath = TmpDirectoryPath / Id;

  try {
    fs::create_directories(NewDirectoryPath);
    fs::permissions(NewDirectoryPath, fs::perms::owner_read |
                                          fs::perms::owner_write |
                                          fs::perms::owner_exec); // 0700

  } catch (const fs::filesystem_error &E) {
    throw sycl::exception(sycl::errc::build, E.what());
  }

  return NewDirectoryPath;
}

void deleteWS(const fs::path &ParentDir) {
  try {
    fs::remove_all(ParentDir);
  } catch (const fs::filesystem_error &E) {
    // We could simply suppress this, since deleting the directory afterwards
    // is not critical. But if there are problems, seems good to know.
    throw sycl::exception(sycl::errc::build, E.what());
  }
}

std::string userArgsAsString(const std::vector<std::string> &UserArguments) {
  return std::accumulate(UserArguments.begin(), UserArguments.end(),
                         std::string(""),
                         [](const std::string &A, const std::string &B) {
                           return A.empty() ? B : A + " " + B;
                         });
}

void outputPreamble(std::ofstream &Os, const std::string &Id,
                    const std::vector<std::string> &UserArgs) {

  Os << "/*\n";
  Os << "  clang++ -fsycl -o " << Id << ".bin ";
  Os << userArgsAsString(UserArgs);
  Os << " -fno-sycl-dead-args-optimization -fsycl-dump-device-code=./ " << Id;
  Os << ".cpp \n */" << std::endl;
}

fs::path outputCpp(const fs::path &ParentDir, const std::string &Id,
                   std::string RawCodeString,
                   const std::vector<std::string> &UserArgs,
                   const std::vector<std::string> &RegisteredKernelNames) {
  fs::path FilePath = ParentDir / (Id + ".cpp");
  std::ofstream Outfile(FilePath, std::ios::out | std::ios::trunc);

  if (Outfile.is_open()) {
    outputPreamble(Outfile, Id, UserArgs);
    Outfile << RawCodeString << std::endl;

    // Temporarily needed until -c works with -fsycl-dump-spirv.
    Outfile << "int main() {\n";
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

void outputIncludeFiles(const fs::path &Dirpath, include_pairs_t IncludePairs) {
  using pairStrings = std::pair<std::string, std::string>;
  for (pairStrings p : IncludePairs) {
    fs::path FilePath = Dirpath / p.first;
    fs::create_directories(FilePath.parent_path());
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
#ifdef _WIN32
  std::string Compiler = "clang++.exe";
#else
  std::string Compiler = "clang++";
#endif
  return Compiler;
}

// We are assuming that the compiler is in /bin and the shared lib in
// the adjacent /lib.
fs::path getCompilerPath() {
  std::string Compiler = getCompilerName();
  const std::string LibSYCLDir = sycl::detail::OSUtil::getCurrentDSODir();
  fs::path CompilerPath = fs::path(LibSYCLDir) / ".." / "bin" / Compiler;
  return CompilerPath;
}

int invokeCommand(const std::string &command, std::string &output) {
#ifdef _WIN32
  FILE *pipe = _popen(command.c_str(), "r");
#else
  FILE *pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) {
    return -1;
  }

  char buffer[1024];
  while (!feof(pipe)) {
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
      output += buffer;
    }
  }

#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif

  return 0;
}

std::string invokeCompiler(const fs::path &FPath, const fs::path &DPath,
                           const std::string &Id,
                           const std::vector<std::string> &UserArgs,
                           std::string *LogPtr) {

  fs::path FilePath(FPath);
  fs::path ParentDir(DPath);
  fs::path TargetPath = ParentDir / (Id + ".bin");
  fs::path LogPath = ParentDir / "compilation_log.txt";
  std::string Compiler = getCompilerPath().make_preferred().string();

  std::string Command =
      Compiler + " -fsycl -o " + TargetPath.make_preferred().string() + " " +
      userArgsAsString(UserArgs) +
      " -fno-sycl-dead-args-optimization -fsycl-dump-device-code=" +
      ParentDir.make_preferred().string() + " " +
      FilePath.make_preferred().string() + " 2>&1";

  std::string CompileLog;
  int Result = invokeCommand(Command, CompileLog);

  if (LogPtr != nullptr) {
    LogPtr->append(CompileLog);
  }

  // There is little chance of Result being non-zero.
  // Actual compilation failure is not detected by error code,
  // but by missing .spv files.
  if (Result != 0) {
    throw sycl::exception(sycl::errc::build,
                          "Compile failure: " + std::to_string(Result) + " " +
                              CompileLog);
  }
  return CompileLog;
}

fs::path findSpv(const fs::path &ParentDir, const std::string &Id,
                 std::string &CompileLog) {
  std::regex PatternRegex(Id + R"(.*\.spv)");

  // Iterate through all files in the directory matching the pattern.
  for (const auto &Entry : fs::directory_iterator(ParentDir)) {
    if (fs::is_regular_file(Entry.path()) &&
        std::regex_match(Entry.path().filename().string(), PatternRegex)) {
      return Entry.path(); // Return the path if it matches the SPV pattern.
    }
  }

  // Missing .spv file indicates there was a compilation failure.
  throw sycl::exception(sycl::errc::build, "Compile failure: " + CompileLog);
}

spirv_vec_t loadSpvFromFile(const fs::path &FileName) {
  std::ifstream SpvStream(FileName, std::ios::binary);
  SpvStream.seekg(0, std::ios::end);
  size_t Size = SpvStream.tellg();
  SpvStream.seekg(0);
  spirv_vec_t Spv(Size);
  SpvStream.read(reinterpret_cast<char *>(Spv.data()), Size);

  return Spv;
}

spirv_vec_t
SYCL_to_SPIRV(const std::string &SYCLSource, include_pairs_t IncludePairs,
              const std::vector<std::string> &UserArgs, std::string *LogPtr,
              const std::vector<std::string> &RegisteredKernelNames) {
  // clang-format off
  const std::string id      = generateSemiUniqueId();
  const fs::path ParentDir  = prepareWS(id);
  fs::path FilePath         = outputCpp(ParentDir, id, SYCLSource, UserArgs, RegisteredKernelNames);
                              outputIncludeFiles(ParentDir, IncludePairs);
  std::string CompileLog    = invokeCompiler(FilePath, ParentDir, id, UserArgs, LogPtr);
  fs::path SpvPath          = findSpv(ParentDir, id, CompileLog);
  spirv_vec_t Spv           = loadSpvFromFile(SpvPath);
                              deleteWS(ParentDir);
                              return Spv;
  // clang-format on
}

bool SYCL_Compilation_Available() {
  // Is compiler on $PATH ? We try to invoke it.
  std::string id = generateSemiUniqueId();
  const fs::path tmp = fs::temp_directory_path();
  fs::path DumpPath = tmp / (id + "_version.txt");
  std::string Compiler = getCompilerPath().make_preferred().string();
  std::string TestCommand =
      Compiler + " --version > " + DumpPath.make_preferred().string();
  int result = std::system(TestCommand.c_str());

  return (result == 0);
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#if SYCL_EXT_JIT_ENABLE
#include "../jit_compiler.hpp"
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

bool SYCL_JIT_Compilation_Available() {
#if SYCL_EXT_JIT_ENABLE
  return sycl::detail::jit_compiler::get_instance().isAvailable();
#else
  return false;
#endif
}

sycl_device_binaries SYCL_JIT_to_SPIRV(
    [[maybe_unused]] const std::string &SYCLSource,
    [[maybe_unused]] include_pairs_t IncludePairs,
    [[maybe_unused]] const std::vector<std::string> &UserArgs,
    [[maybe_unused]] std::string *LogPtr,
    [[maybe_unused]] const std::vector<std::string> &RegisteredKernelNames) {
#if SYCL_EXT_JIT_ENABLE
  return sycl::detail::jit_compiler::get_instance().compileSYCL(
      "rtc", SYCLSource, IncludePairs, UserArgs, LogPtr, RegisteredKernelNames);
#else
  throw sycl::exception(sycl::errc::build,
                        "kernel_compiler via sycl-jit is not available");
#endif
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
