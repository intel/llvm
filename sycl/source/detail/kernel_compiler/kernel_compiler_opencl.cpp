//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/ur.hpp> // getOsLibraryFuncAddress
#include <sycl/exception.hpp> // make_error_code

#include "kernel_compiler_opencl.hpp"

#include "../split_string.hpp"
#include "ocloc_api.h"

#include <cstring>    // strlen
#include <functional> // for std::function
#include <numeric>    // for std::accumulate
#include <regex>
#include <sstream>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// forward declaration
std::string InvokeOclocQuery(const std::vector<uint32_t> &IPVersionVec,
                             const char *identifier);

// ensures the OclocLibrary has the right version, etc.
void checkOclocLibrary(void *OclocLibrary) {
  void *OclocVersionHandle =
      sycl::detail::ur::getOsLibraryFuncAddress(OclocLibrary, "oclocVersion");
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

static std::unique_ptr<void, std::function<void(void *)>>
    OclocLibrary(nullptr, [](void *StoredPtr) {
      if (!StoredPtr)
        return;
      std::ignore = sycl::detail::ur::unloadOsLibrary(StoredPtr);
    });

void loadOclocLibrary(const std::vector<uint32_t> &IPVersionVec) {
#ifdef __SYCL_RT_OS_WINDOWS
  // first the environment, if not compatible will move on to absolute path.
  static const std::vector<std::string_view> OclocPaths = {
      "ocloc64.dll",
      "C:\\Program Files (x86)\\Intel\\oneAPI\\ocloc\\latest\\ocloc64.dll"};
#else
  static const std::vector<std::string_view> OclocPaths = {"libocloc.so"};
#endif

  // attemptLoad() sets OclocLibrary value by side effect.
  auto attemptLoad = [&](std::string_view OclocPath_sv) {
    std::string OclocPath(OclocPath_sv);
    try {
      // Load then perform checks. Each check throws.
      void *tempPtr = sycl::detail::ur::loadOsLibrary(OclocPath);
      OclocLibrary.reset(tempPtr);

      if (tempPtr == nullptr)
        throw sycl::exception(make_error_code(errc::build),
                              "Unable to load ocloc from " + OclocPath);

      checkOclocLibrary(tempPtr);

      InvokeOclocQuery(IPVersionVec, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    } catch (const sycl::exception &) {
      OclocLibrary.reset(nullptr);
      return false;
    }
    return true;
  };
  for (const std::string_view &result : OclocPaths) {
    if (attemptLoad(result))
      return; // exit on successful attempt
  }
  // If we haven't exited yet, then throw to indicate failure.
  throw sycl::exception(make_error_code(errc::build), "Unable to load ocloc");
}

bool OpenCLC_Compilation_Available(const std::vector<uint32_t> &IPVersionVec) {
  // Already loaded?
  if (OclocLibrary != nullptr)
    return true;

  try {
    // loads and checks version
    loadOclocLibrary(IPVersionVec);
    return true;
  } catch (...) {
    return false;
  }
}

using voidPtr = void *;

void SetupLibrary(voidPtr &oclocInvokeHandle, voidPtr &oclocFreeOutputHandle,
                  std::error_code the_errc,
                  const std::vector<uint32_t> &IPVersionVec) {
  if (OclocLibrary == nullptr)
    loadOclocLibrary(IPVersionVec);

  if (!oclocInvokeHandle) {
    oclocInvokeHandle = sycl::detail::ur::getOsLibraryFuncAddress(
        OclocLibrary.get(), "oclocInvoke");
    if (!oclocInvokeHandle)
      throw sycl::exception(the_errc, "Cannot load oclocInvoke() function");

    oclocFreeOutputHandle = sycl::detail::ur::getOsLibraryFuncAddress(
        OclocLibrary.get(), "oclocFreeOutput");
    if (!oclocFreeOutputHandle)
      throw sycl::exception(the_errc, "Cannot load oclocFreeOutput() function");
  }
}

std::string IPVersionsToString(const std::vector<uint32_t> IPVersionVec) {
  std::stringstream ss;
  ss.imbue(std::locale::classic());
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

std::string InvokeOclocQuery(const std::vector<uint32_t> &IPVersionVec,
                             const char *identifier) {

  std::string QueryLog = "";

  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code the_errc = make_error_code(errc::runtime);

  SetupLibrary(oclocInvokeHandle, oclocFreeOutputHandle, the_errc,
               IPVersionVec);

  uint32_t NumOutputs = 0;
  uint8_t **Outputs = nullptr;
  uint64_t *OutputLengths = nullptr;
  char **OutputNames = nullptr;

  std::vector<const char *> Args = {"ocloc", "query"};
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
      if (OutputLengths[i] > 0) {
        const char *LogText = reinterpret_cast<const char *>(Outputs[i]);
        QueryLog.append(LogText, OutputLengths[i]);
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

spirv_vec_t
OpenCLC_to_SPIRV(const std::string &Source,
                 const std::vector<uint32_t> &IPVersionVec,
                 const std::vector<sycl::detail::string_view> &UserArgs,
                 std::string *LogPtr) {
  // handles into ocloc shared lib
  static void *oclocInvokeHandle = nullptr;
  static void *oclocFreeOutputHandle = nullptr;
  std::error_code build_errc = make_error_code(errc::build);

  SetupLibrary(oclocInvokeHandle, oclocFreeOutputHandle, build_errc,
               IPVersionVec);

  // assemble ocloc args
  std::string CombinedUserArgs = "";
  for (const sycl::detail::string_view &UserArg : UserArgs) {
    CombinedUserArgs += UserArg.data();
    CombinedUserArgs += " ";
  }

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

  std::string IPVersionsStr;
  std::string OpenCLCFeaturesOption;
  std::string ExtensionsOption;
  std::string VersionOption;
  auto hasSingleDeviceOrSameDevices = [](auto &IPVersionVec) -> bool {
    auto IPVersion = IPVersionVec.begin();
    for (auto IPVersionItem = ++std::begin(IPVersionVec);
         IPVersionItem != std::end(IPVersionVec); IPVersionItem++)
      if (*IPVersionItem != *IPVersion)
        return false;

    return true;
  };

  assert(IPVersionVec.size() >= 1 &&
         "At least one device must be provided to build_from_source(...).");
  if (hasSingleDeviceOrSameDevices(IPVersionVec)) {
    // If we have a single device (or all devices are the same) then pass it
    // through -device option to enable all extensions supported by that device.
    IPVersionsStr = IPVersionsToString({IPVersionVec.at(0)});
    if (!IPVersionsStr.empty()) {
      Args.push_back("-device");
      Args.push_back(IPVersionsStr.c_str());
    }
  } else {
    // Currently ocloc -spv_only doesn't produce spirv file when multiple
    // devices are provided via -device option. That's why in this case we have
    // to enable common extensions supported by all devices manually.

    // Find maximum opencl version supported by all devices in IPVersionVec.
    auto OpenCLVersions =
        InvokeOclocQuery(IPVersionVec, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    const std::regex VersionRegEx("[0-9].[0-9].[0-9]");
    std::string const &(*max)(std::string const &, std::string const &) =
        std::max<std::string>;
    auto MaxVersion = std::accumulate(
        std::sregex_token_iterator(OpenCLVersions.begin(), OpenCLVersions.end(),
                                   VersionRegEx),
        std::sregex_token_iterator(), std::string("0.0.0"), max);

    // Find common extensions supported by all devices in IPVersionVec.
    // Lambda to accumulate extensions in the format +extension1,+extension2...
    // to pass to ocloc as an option.
    auto Accum = [](const std::string &acc, const std::string &s) {
      return acc + (acc.empty() ? "+" : ",+") + s;
    };

    // If OpenCL version is higher that 3.0.0 then we need to enable OpenCL C
    // features as well in addition to CL extensions.
    if (MaxVersion >= "3.0.0") {
      // construct a string which enables common extensions supported by
      // devices.
      auto OpenCLCFeatures =
          InvokeOclocQuery(IPVersionVec, "CL_DEVICE_OPENCL_C_FEATURES");
      const std::regex OpenCLCRegEx("__opencl_c_[^:]+");
      auto OpenCLCFeaturesValue = std::accumulate(
          std::sregex_token_iterator(OpenCLCFeatures.begin(),
                                     OpenCLCFeatures.end(), OpenCLCRegEx),
          std::sregex_token_iterator(), std::string(""), Accum);
      if (OpenCLCFeaturesValue.size()) {
        OpenCLCFeaturesOption = "-cl-ext=" + OpenCLCFeaturesValue;
        Args.push_back("-internal_options");
        Args.push_back(OpenCLCFeaturesOption.c_str());
      }
    }

    // Accumulate CL extensions into an option.
    auto Extensions = InvokeOclocQuery(IPVersionVec, "CL_DEVICE_EXTENSIONS");
    const std::regex CLRegEx("cl_[^\\s]+");
    auto ExtensionsValue =
        std::accumulate(std::sregex_token_iterator(Extensions.begin(),
                                                   Extensions.end(), CLRegEx),
                        std::sregex_token_iterator(), std::string(""), Accum);
    if (ExtensionsValue.size()) {
      ExtensionsOption = "-cl-ext=" + ExtensionsValue;
      Args.push_back("-internal_options");
      Args.push_back(ExtensionsOption.c_str());
    }
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
      if (OutputLengths[i] > 0) {
        const char *LogText = reinterpret_cast<const char *>(Outputs[i]);
        CompileLog.append(LogText, OutputLengths[i]);
        if (LogPtr != nullptr)
          LogPtr->append(LogText, OutputLengths[i]);
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

bool OpenCLC_Feature_Available(const std::string &Feature, uint32_t IPVersion) {
  static std::string FeatureLog = "";
  if (FeatureLog.empty()) {
    try {
      FeatureLog = InvokeOclocQuery({IPVersion}, "CL_DEVICE_OPENCL_C_FEATURES");
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
          InvokeOclocQuery({IPVersion}, "CL_DEVICE_OPENCL_C_ALL_VERSIONS");
    } catch (sycl::exception &) {
      return false;
    }
  }

  // Have VersionLog, will search.
  // "OpenCL C":1.0.0 "OpenCL C":1.1.0 "OpenCL C":1.2.0 "OpenCL C":3.0.0
  std::stringstream ss;
  ss << Version.major << "." << Version.minor << "." << Version.patch;
  return VersionLog.find(ss.str()) != std::string::npos;
}

bool OpenCLC_Supports_Extension(
    const std::string &Name, ext::oneapi::experimental::cl_version *VersionPtr,
    uint32_t IPVersion) {
  std::error_code rt_errc = make_error_code(errc::runtime);
  static std::string ExtensionByVersionLog = "";
  if (ExtensionByVersionLog.empty()) {
    try {
      ExtensionByVersionLog =
          InvokeOclocQuery({IPVersion}, "CL_DEVICE_EXTENSIONS_WITH_VERSION");
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

  // Note that VersionPtr is an optional parameter in
  // ext_oneapi_supports_cl_extension().
  if (!VersionPtr)
    return true;

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
    std::string result = InvokeOclocQuery({IPVersion}, "CL_DEVICE_PROFILE");
    // NOTE: result has \n\n amended. Clean it up.
    // TODO: remove this once the ocloc query is fixed.
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](char c) {
                                  return !std::isprint(c) || std::isspace(c);
                                }),
                 result.end());

    return result;
  } catch (sycl::exception &) {
    return "";
  }
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
