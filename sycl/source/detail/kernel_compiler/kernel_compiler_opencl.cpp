//==-- kernel_compiler_ opencl.cpp  OpenCL kernel compilation support      -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/pi.hpp>                         // getOsLibraryFuncAddress
#include <sycl/exception.hpp>                         // for make_error_code

#include "kernel_compiler_opencl.hpp"

#include "../online_compiler/ocloc_api.h"

#include <cstring> // strlen
#include <numeric> // for std::accumulate

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// copy/pasta from online_compiler.cpp
// ensures the OclocLibrary has the right version, etc.
void checkOclocLibrary(void *OclocLibrary) {
  void *OclocVersionHandle =
      sycl::detail::pi::getOsLibraryFuncAddress(OclocLibrary, "oclocVersion");
  // The initial versions of ocloc library did not have the oclocVersion()
  // function. Those versions had the same API as the first version of ocloc
  // library having that oclocVersion() function.
  int LoadedVersion = ocloc_version_t::OCLOC_VERSION_1_0;
  if (OclocVersionHandle) {
    decltype(::oclocVersion) *OclocVersionFunc =
        reinterpret_cast<decltype(::oclocVersion) *>(OclocVersionHandle);
    LoadedVersion = OclocVersionFunc();
  }
  // The loaded library with version (A.B) is compatible with expected API/ABI
  // version (X.Y) used here if A == B and B >= Y.
  int LoadedVersionMajor = LoadedVersion >> 16;
  int LoadedVersionMinor = LoadedVersion & 0xffff;
  int CurrentVersionMajor = ocloc_version_t::OCLOC_VERSION_CURRENT >> 16;
  int CurrentVersionMinor = ocloc_version_t::OCLOC_VERSION_CURRENT & 0xffff;
  if (LoadedVersionMajor != CurrentVersionMajor ||
      LoadedVersionMinor < CurrentVersionMinor) {
    throw sycl::exception(
        make_error_code(errc::build),
        std::string("Found incompatible version of ocloc library: (") +
            std::to_string(LoadedVersionMajor) + "." +
            std::to_string(LoadedVersionMinor) +
            "). The supported versions are (" +
            std::to_string(CurrentVersionMajor) +
            ".N), where (N >= " + std::to_string(CurrentVersionMinor) + ").");
  }
}

// load the ocloc shared library, check it.
void *loadOclocLibrary() {
#ifdef __SYCL_RT_OS_WINDOWS
  static const std::string OclocLibraryName = "ocloc64.dll";
#else
  static const std::string OclocLibraryName = "libocloc.so";
#endif
  void *OclocLibrary = sycl::detail::pi::loadOsLibrary(OclocLibraryName);
  if (!OclocLibrary)
    throw sycl::exception(make_error_code(errc::build),
                          "Unable to load ocloc library " + OclocLibraryName);

  checkOclocLibrary(OclocLibrary);

  return OclocLibrary;
}

spirv_vec_t OpenCLC_to_SPIRV(const std::string &Source, const std::vector<std::string> &UserArgs) {
  std::vector<std::string> CMUserArgs = UserArgs;
  CMUserArgs.push_back("-cmc");

  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code build_errc = make_error_code(errc::build);

  // setup Library
  if (!oclocInvokeHandle) {
    void *OclocLibrary = loadOclocLibrary();

    oclocInvokeHandle =
        sycl::detail::pi::getOsLibraryFuncAddress(OclocLibrary, "oclocInvoke");
    if (!oclocInvokeHandle)
      throw sycl::exception(build_errc, "Cannot load oclocInvoke() function");

    oclocFreeOutputHandle = sycl::detail::pi::getOsLibraryFuncAddress(
        OclocLibrary, "oclocFreeOutput");
    if (!oclocFreeOutputHandle)
      throw sycl::exception(build_errc,
                            "Cannot load oclocFreeOutput() function");
  }

  // assemble ocloc args
  std::string CombinedUserArgs =
      std::accumulate(UserArgs.begin(), UserArgs.end(), std::string(""),
                      [](const std::string &acc, const std::string &s) {
                        return acc + s + " ";
                      });

  std::vector<const char *> Args = {"ocloc", "-q", "-spv_only", "-options",
                                    CombinedUserArgs.c_str()};

  uint32_t NumOutputs = 0;
  //std::byte **Outputs = nullptr;
  uint8_t **Outputs = nullptr;
  uint64_t *OutputLengths = nullptr;
  char **OutputNames = nullptr;

  //const std::byte *Sources[] = {reinterpret_cast<const std::byte *>(Source.c_str())};
  const uint8_t *Sources[] = {reinterpret_cast<const uint8_t *>(Source.c_str())};
  const char *SourceName = "main.cl";
  const uint64_t SourceLengths[] = {Source.length() + 1};

  Args.push_back("-file");
  Args.push_back(SourceName);

  // invoke
  decltype(::oclocInvoke) *OclocInvokeFunc =
      reinterpret_cast<decltype(::oclocInvoke) *>(oclocInvokeHandle);
  int CompileError =
      OclocInvokeFunc(Args.size(), Args.data(), 1, Sources, SourceLengths,
                      &SourceName, 0, nullptr, nullptr, nullptr, &NumOutputs,
                      &Outputs, &OutputLengths, &OutputNames);

  // gather the results ( the SpirV and the Log)
  spirv_vec_t SpirV;
  std::string CompileLog;
  for (uint32_t i = 0; i < NumOutputs; i++) {
    size_t NameLen = strlen(OutputNames[i]);
    if (NameLen >= 4 && strstr(OutputNames[i], ".spv") != nullptr &&
        Outputs[i] != nullptr) {
      assert(SpirV.size() == 0 && "More than one SPIR-V output found.");
      SpirV = spirv_vec_t(Outputs[i], Outputs[i] + OutputLengths[i]);
    } else if (!strcmp(OutputNames[i], "stdout.log")) {
      CompileLog = std::string(reinterpret_cast<const char *>(Outputs[i]));
    }
  }
  // std::cout << "Compile Log: " << std::endl << CompileLog << std::endl <<
  // "=============" << std::endl;

  // Try to free memory before reporting possible error.
  decltype(::oclocFreeOutput) *OclocFreeOutputFunc =
      reinterpret_cast<decltype(::oclocFreeOutput) *>(oclocFreeOutputHandle);
  int MemFreeError =
      OclocFreeOutputFunc(&NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  if (CompileError)
    throw sycl::exception(build_errc, "ocloc reported compilation errors: {\n" +
                                          CompileLog + "\n}");

  if (SpirV.empty())
    throw sycl::exception(build_errc,
                          "Unexpected output: ocloc did not return SPIR-V");

  if (MemFreeError)
    throw sycl::exception(build_errc, "ocloc cannot safely free resources");

  return SpirV;
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl