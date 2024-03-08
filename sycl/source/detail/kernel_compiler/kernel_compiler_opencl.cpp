//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common_info.hpp> // split_string
#include <sycl/detail/pi.hpp>          // getOsLibraryFuncAddress
#include <sycl/exception.hpp>          // make_error_code

#include "kernel_compiler_opencl.hpp"

#include "../online_compiler/ocloc_api.h"

#include <cstring> // strlen
#include <numeric> // for std::accumulate

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

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

static void *OclocLibrary = nullptr;

// load the ocloc shared library, check it.
void *loadOclocLibrary() {
#ifdef __SYCL_RT_OS_WINDOWS
  static const std::string OclocLibraryName = "ocloc64.dll";
#else
  static const std::string OclocLibraryName = "libocloc.so";
#endif
  void *tempPtr = OclocLibrary;
  if (tempPtr == nullptr) {
    tempPtr = sycl::detail::pi::loadOsLibrary(OclocLibraryName);

    if (tempPtr == nullptr)
      throw sycl::exception(make_error_code(errc::build),
                            "Unable to load ocloc library " + OclocLibraryName);

    checkOclocLibrary(tempPtr);

    OclocLibrary = tempPtr;
  }

  return OclocLibrary;
}

bool OpenCLC_Compilation_Available() {
  // Already loaded?
  if (OclocLibrary != nullptr)
    return true;

  try {
    // loads and checks version
    loadOclocLibrary();
    return true;
  } catch (...) {
    return false;
  }
}

using voidPtr = void *;

void SetupLibrary(voidPtr &oclocInvokeHandle, voidPtr &oclocFreeOutputHandle,
                  std::error_code the_errc) {
  if (!oclocInvokeHandle) {
    if (OclocLibrary == nullptr)
      loadOclocLibrary();

    oclocInvokeHandle =
        sycl::detail::pi::getOsLibraryFuncAddress(OclocLibrary, "oclocInvoke");
    if (!oclocInvokeHandle)
      throw sycl::exception(the_errc, "Cannot load oclocInvoke() function");

    oclocFreeOutputHandle = sycl::detail::pi::getOsLibraryFuncAddress(
        OclocLibrary, "oclocFreeOutput");
    if (!oclocFreeOutputHandle)
      throw sycl::exception(the_errc, "Cannot load oclocFreeOutput() function");
  }
}

std::string IPVersionsToString(const std::vector<uint32_t> IPVersionVec) {
  std::stringstream ss;
  bool amFirst = true;
  for (uint32_t ipVersion : IPVersionVec) {
    // if any device is not intelGPU, bail.
    if (ipVersion < 0x02000000)
      return "";

    if (!amFirst)
      ss << ",";
    amFirst = false;
    ss << ipVersion;
  }
  return ss.str();
}

spirv_vec_t OpenCLC_to_SPIRV(const std::string &Source,
                             const std::vector<uint32_t> &IPVersionVec,
                             const std::vector<std::string> &UserArgs,
                             std::string *LogPtr) {
  std::vector<std::string> CMUserArgs = UserArgs;
  CMUserArgs.push_back("-cmc");

  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code build_errc = make_error_code(errc::build);

  SetupLibrary(oclocInvokeHandle, oclocFreeOutputHandle, build_errc);

  // assemble ocloc args
  std::string CombinedUserArgs =
      std::accumulate(UserArgs.begin(), UserArgs.end(), std::string(""),
                      [](const std::string &acc, const std::string &s) {
                        return acc + s + " ";
                      });

  std::vector<const char *> Args = {"ocloc", "-q", "-spv_only", "-options",
                                    CombinedUserArgs.c_str()};

  uint32_t NumOutputs = 0;
  uint8_t **Outputs = nullptr;
  uint64_t *OutputLengths = nullptr;
  char **OutputNames = nullptr;

  const uint8_t *Sources[] = {
      reinterpret_cast<const uint8_t *>(Source.c_str())};
  const char *SourceName = "main.cl";
  const uint64_t SourceLengths[] = {Source.length() + 1};

  Args.push_back("-file");
  Args.push_back(SourceName);

  // device
  std::string IPVersionsStr = IPVersionsToString(IPVersionVec);
  if (!IPVersionsStr.empty()) {
    Args.push_back("-device");
    Args.push_back(IPVersionsStr.c_str());
  }

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
      const char *LogText = reinterpret_cast<const char *>(Outputs[i]);
      if (LogText != nullptr && LogText[0] != '\0') {
        CompileLog.append(LogText);
        if (LogPtr != nullptr)
          LogPtr->append(LogText);
      }
    }
  }

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

std::string InvokeOclocQuery(uint32_t IPVersion, const char *identifier) {

  std::string QueryLog = "";

  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code the_errc = make_error_code(errc::runtime);

  SetupLibrary(oclocInvokeHandle, oclocFreeOutputHandle, the_errc);

  uint32_t NumOutputs = 0;
  uint8_t **Outputs = nullptr;
  uint64_t *OutputLengths = nullptr;
  char **OutputNames = nullptr;

  std::vector<const char *> Args = {"ocloc", "query"};
  std::vector<uint32_t> IPVersionVec{IPVersion};
  std::string IPVersionsStr = IPVersionsToString(IPVersionVec);
  if (!IPVersionsStr.empty()) {
    Args.push_back("-device");
    Args.push_back(IPVersionsStr.c_str());
  }
  Args.push_back(identifier);

  decltype(::oclocInvoke) *OclocInvokeFunc =
      reinterpret_cast<decltype(::oclocInvoke) *>(oclocInvokeHandle);

  int InvokeError = OclocInvokeFunc(
      Args.size(), Args.data(), 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
      nullptr, &NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  // Gather the results.
  for (uint32_t i = 0; i < NumOutputs; i++) {
    if (!strcmp(OutputNames[i], "stdout.log")) {
      const char *LogText = reinterpret_cast<const char *>(Outputs[i]);
      if (LogText != nullptr && LogText[0] != '\0') {
        QueryLog.append(LogText);
      }
    }
  }

  // Try to free memory before reporting possible error.
  decltype(::oclocFreeOutput) *OclocFreeOutputFunc =
      reinterpret_cast<decltype(::oclocFreeOutput) *>(oclocFreeOutputHandle);
  int MemFreeError =
      OclocFreeOutputFunc(&NumOutputs, &Outputs, &OutputLengths, &OutputNames);

  if (InvokeError)
    throw sycl::exception(the_errc,
                          "ocloc reported errors: {\n" + QueryLog + "\n}");

  if (MemFreeError)
    throw sycl::exception(the_errc, "ocloc cannot safely free resources");

  return QueryLog;
}

bool OpenCLC_Feature_Available(const std::string &Feature, uint32_t IPVersion) {
  static std::string FeatureLog = "";
  if (FeatureLog.empty()) {
    try {
      FeatureLog = InvokeOclocQuery(IPVersion, "CL_DEVICE_OPENCL_C_FEATURES");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // Allright, we have FeatureLog, so let's find that feature!
  return (FeatureLog.find(Feature) != std::string::npos);
}

bool OpenCLC_Supports_Version(
    const ext::oneapi::experimental::cl_version &Version, uint32_t IPVersion) {
  static std::string VersionLog = "";
  if (VersionLog.empty()) {
    try {
      VersionLog =
          InvokeOclocQuery(IPVersion, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // Have VersionLog, will search.
  // "OpenCL C":1.0.0 "OpenCL C":1.1.0 "OpenCL C":1.2.0 "OpenCL C":3.0.0
  std::stringstream ss;
  ss << Version.major << "." << Version.minor << "." << Version.patch;
  return VersionLog.find(ss.str());
}

bool OpenCLC_Supports_Extension(
    const std::string &Name, ext::oneapi::experimental::cl_version *VersionPtr,
    uint32_t IPVersion) {
  std::error_code rt_errc = make_error_code(errc::runtime);
  static std::string ExtensionByVersionLog = "";
  if (ExtensionByVersionLog.empty()) {
    try {
      ExtensionByVersionLog =
          InvokeOclocQuery(IPVersion, "CL_DEVICE_EXTENSIONS_WITH_VERSION");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // ExtensionByVersionLog is ready. Time to find Name, and update VersionPtr.
  // cl_khr_byte_addressable_store:1.0.0 cl_khr_device_uuid:1.0.0 ...
  size_t where = ExtensionByVersionLog.find(Name);
  if (where == std::string::npos) {
    return false;
  } // not there

  size_t colon = ExtensionByVersionLog.find(':', where);
  if (colon == std::string::npos) {
    throw sycl::exception(
        rt_errc,
        "trouble parsing query returned from CL_DEVICE_EXTENSIONS_WITH_VERSION "
        "- extension not followed by colon (:)");
  }
  colon++; // move it forward

  size_t space = ExtensionByVersionLog.find(' ', colon); // could be npos

  size_t count = (space == std::string::npos) ? space : (space - colon);

  std::string versionStr = ExtensionByVersionLog.substr(colon, count);
  std::vector<std::string> versionVec =
      sycl::detail::split_string(versionStr, '.');
  if (versionVec.size() != 3) {
    throw sycl::exception(
        rt_errc,
        "trouble parsing query returned from  "
        "CL_DEVICE_EXTENSIONS_WITH_VERSION - version string unexpected: " +
            versionStr);
  }

  VersionPtr->major = std::stoi(versionVec[0]);
  VersionPtr->minor = std::stoi(versionVec[1]);
  VersionPtr->patch = std::stoi(versionVec[2]);

  return true;
}

std::string OpenCLC_Profile(uint32_t IPVersion) {
  try {
    std::string result = InvokeOclocQuery(IPVersion, "CL_DEVICE_PROFILE");
    // NOTE: result has \n\n amended. Clean it up.
    // TODO: remove this once the ocloc query is fixed.
    while (result.back() == '\n') {
      result.pop_back();
    }

    return result;
  } catch (sycl::exception &) {
    return "";
  }
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
