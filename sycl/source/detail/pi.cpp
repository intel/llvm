//===-- pi.cpp - PI utilities implementation -------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <cstdarg>
#include <cstring>
#include <iostream>
#include <map>
#include <stddef.h>
#include <string>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {
namespace pi {

std::string platformInfoToString(pi_platform_info info) {
  switch (info) {
  case PI_PLATFORM_INFO_PROFILE:
    return "PI_PLATFORM_INFO_PROFILE";
  case PI_PLATFORM_INFO_VERSION:
    return "PI_PLATFORM_INFO_VERSION";
  case PI_PLATFORM_INFO_NAME:
    return "PI_PLATFORM_INFO_NAME";
  case PI_PLATFORM_INFO_VENDOR:
    return "PI_PLATFORM_INFO_VENDOR";
  case PI_PLATFORM_INFO_EXTENSIONS:
    return "PI_PLATFORM_INFO_EXTENSIONS";
  default:
    die("Unknown pi_platform_info value passed to "
        "cl::sycl::detail::pi::platformInfoToString");
  }
}

// Check for manually selected BE at run-time.
bool useBackend(Backend TheBackend) {
  static const char *GetEnv = std::getenv("SYCL_BE");
  // Current default backend as SYCL_BE_PI_OPENCL
  // Valid values of GetEnv are "PI_OPENCL" and "PI_OTHER"
  std::string StringGetEnv = (GetEnv ? GetEnv : "PI_OPENCL");
  static const Backend Use =
      (StringGetEnv == "PI_OTHER" ? SYCL_BE_PI_OTHER : SYCL_BE_PI_OPENCL);
  return TheBackend == Use;
}

// TODO: Move this global structure into sycl::platform object,
// associate each plugin with a platform.
pi_plugin PluginInformation;

// Find the plugin at the appropriate location and return the location.
// TODO: Change the function appropriately when there are multiple plugins.
std::string findPlugin() {
  // TODO: Based on final design discussions, change the location where the
  // plugin must be searched; how to identify the plugins etc. Currently the
  // search is done for libpi_opencl.so/pi_opencl.dll file in LD_LIBRARY_PATH
  // env only.
  return PLUGIN_NAME;
}

// Load the Plugin by calling the OS dependent library loading call.
// Return the handle to the Library.
void *loadPlugin(const std::string &PluginPath) {
  return loadOsLibrary(PluginPath);
}

// Binds all the PI Interface APIs to Plugin Library Function Addresses.
// TODO: Remove the 'OclPtr' extension to PI_API.
// TODO: Change the functionality such that a single getOsLibraryFuncAddress
// call is done to get all Interface API mapping. The plugin interface also
// needs to setup infrastructure to route PI_CALLs to the appropriate plugins.
// Currently, we bind to a singe plugin.
bool bindPlugin(void *Library) {

  decltype(::piPluginInit) *PluginInitializeFunction = (decltype(
      &::piPluginInit))(getOsLibraryFuncAddress(Library, "piPluginInit"));
  if (PluginInitializeFunction == nullptr)
    return false;

  int err = PluginInitializeFunction(&PluginInformation);

  // TODO: Compare Supported versions and check for backward compatibility.
  // Make sure err is PI_SUCCESS.
  assert((err == PI_SUCCESS) && "Unexpected error when binding to Plugin.");
  (void)err;

  // TODO: Return a more meaningful value/enum.
  return true;
}

// Load the plugin based on SYCL_BE.
// TODO: Currently only accepting OpenCL plugins. Edit it to identify and load
// other kinds of plugins, do the required changes in the findPlugin, loadPlugin
// and bindPlugin functions.
void initialize() {
  static bool Initialized = false;
  if (Initialized) {
    return;
  }
  if (!useBackend(SYCL_BE_PI_OPENCL)) {
    die("Unknown SYCL_BE");
  }

  std::string PluginPath = findPlugin();
  if (PluginPath.empty())
    die("Plugin Not Found.");

  void *Library = loadPlugin(PluginPath);
  if (!Library) {
    std::string Message =
        "Check if plugin is present. Failed to load plugin: " + PluginPath;
    die(Message.c_str());
  }

  if (!bindPlugin(Library)) {
    std::string Message = "Failed to bind PI APIs to the plugin: " + PluginPath;
    die(Message.c_str());
  }

  Initialized = true;
}

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message) {
  std::cerr << "pi_die: " << Message << std::endl;
  std::terminate();
}

void assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

// TODO: Pass platform object to constructor which will contain the
// PluginInformation class. Platform class with Plugin information is not
// implemented yet.

#define _PI_API(api)                                                           \
  template <>                                                                  \
  CallPi<decltype(&::api),                                                     \
         (offsetof(pi_plugin::FunctionPointers, api))>::CallPi() {             \
    initialize();                                                              \
    MFnPtr = (RT::PluginInformation.PiFunctionTable.api);                      \
    MFnName = #api;                                                            \
  }
#include <CL/sycl/detail/pi.def>

} // namespace pi
} // namespace detail
} // namespace sycl
} // namespace cl
