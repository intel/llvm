//===-- pi.cpp - PI utilities implementation -------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.cpp
/// Implementation of C++ wrappers for PI interface.
///
/// \ingroup sycl_pi

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>

#include <bitset>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <map>
#include <stddef.h>
#include <string>
#include <sstream>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
// Stream name being used for traces generated from the SYCL runtime
constexpr const char *PICALL_STREAM_NAME = "sycl.pi";
// Global (to the SYCL runtime) graph handle that all command groups are a
// child of
///< Event to be used by graph related activities
xpti_td *GSYCLGraphEvent = nullptr;
///< Event to be used by PI layer related activities
xpti_td *GPICallEvent = nullptr;
///< Constansts being used as placeholder until one is able to reliably get the
///< version of the SYCL runtime
constexpr uint32_t GMajVer = 1;
constexpr uint32_t GMinVer = 0;
constexpr const char *GVerStr = "sycl 1.0";
#endif

namespace pi {

bool XPTIInitDone = false;

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

std::string memFlagToString(pi_mem_flags Flag) {
  assertion(((Flag == 0u) || ((Flag & (Flag - 1)) == 0)) &&
            "More than one bit set");

  std::stringstream Sstream;

  switch (Flag) {
  case pi_mem_flags{0}:
    Sstream << "pi_mem_flags(0)";
    break;
  case PI_MEM_FLAGS_ACCESS_RW:
    Sstream << "PI_MEM_FLAGS_ACCESS_RW";
    break;
  case PI_MEM_FLAGS_HOST_PTR_USE:
    Sstream << "PI_MEM_FLAGS_HOST_PTR_USE";
    break;
  case PI_MEM_FLAGS_HOST_PTR_COPY:
    Sstream << "PI_MEM_FLAGS_HOST_PTR_COPY";
    break;
  default:
    Sstream << "unknown pi_mem_flags bit == " << Flag;
  }

  return Sstream.str();
}

std::string memFlagsToString(pi_mem_flags Flags) {
  std::stringstream Sstream;
  bool FoundFlag = false;

  auto FlagSeparator = [](bool FoundFlag) { return FoundFlag ? "|" : ""; };

  pi_mem_flags ValidFlags[] = {PI_MEM_FLAGS_ACCESS_RW,
                               PI_MEM_FLAGS_HOST_PTR_USE,
                               PI_MEM_FLAGS_HOST_PTR_COPY};

  if (Flags == 0u) {
    Sstream << "pi_mem_flags(0)";
  } else {
    for (const auto Flag : ValidFlags) {
      if (Flag & Flags) {
        Sstream << FlagSeparator(FoundFlag) << memFlagToString(Flag);
        FoundFlag = true;
      }
    }

    std::bitset<64> UnkownBits(Flags & ~(PI_MEM_FLAGS_ACCESS_RW |
                                         PI_MEM_FLAGS_HOST_PTR_USE |
                                         PI_MEM_FLAGS_HOST_PTR_COPY));
    if (UnkownBits.any()) {
      Sstream << FlagSeparator(FoundFlag)
              << "unknown pi_mem_flags bits == " << UnkownBits;
    }
  }

  return Sstream.str();
}

// Check for manually selected BE at run-time.
static Backend getBackend() {
  static const char *GetEnv = std::getenv("SYCL_BE");
  // Current default backend as SYCL_BE_PI_OPENCL
  // Valid values of GetEnv are "PI_OPENCL", "PI_CUDA" and "PI_OTHER"
  std::string StringGetEnv = (GetEnv ? GetEnv : "PI_OPENCL");
  static const Backend Use =
    std::map<std::string, Backend>{
      { "PI_OPENCL", SYCL_BE_PI_OPENCL },
      { "PI_CUDA", SYCL_BE_PI_CUDA },
      { "PI_OTHER",  SYCL_BE_PI_OTHER }
    }[ GetEnv ? StringGetEnv : "PI_OPENCL"];
  return Use;
}

// Check for manually selected BE at run-time.
bool useBackend(Backend TheBackend) {
  return TheBackend == getBackend();
}

// GlobalPlugin is a global Plugin used with Interoperability constructors that
// use OpenCL objects to construct SYCL class objects.
std::shared_ptr<plugin> GlobalPlugin;

// Find the plugin at the appropriate location and return the location.
// TODO: Change the function appropriately when there are multiple plugins.
bool findPlugins(vector_class<std::string> &PluginNames) {
  // TODO: Based on final design discussions, change the location where the
  // plugin must be searched; how to identify the plugins etc. Currently the
  // search is done for libpi_opencl.so/pi_opencl.dll file in LD_LIBRARY_PATH
  // env only.
  PluginNames.push_back(OPENCL_PLUGIN_NAME);
  PluginNames.push_back(CUDA_PLUGIN_NAME);
  return true;
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
bool bindPlugin(void *Library, PiPlugin *PluginInformation) {

  decltype(::piPluginInit) *PluginInitializeFunction = (decltype(
      &::piPluginInit))(getOsLibraryFuncAddress(Library, "piPluginInit"));
  if (PluginInitializeFunction == nullptr)
    return false;

  int Err = PluginInitializeFunction(PluginInformation);

  // TODO: Compare Supported versions and check for backward compatibility.
  // Make sure err is PI_SUCCESS.
  assert((Err == PI_SUCCESS) && "Unexpected error when binding to Plugin.");
  (void)Err;

  // TODO: Return a more meaningful value/enum.
  return true;
}

// Load the plugin based on SYCL_BE.
// TODO: Currently only accepting OpenCL and CUDA plugins. Edit it to identify
// and load other kinds of plugins, do the required changes in the
// findPlugins, loadPlugin and bindPlugin functions.
vector_class<plugin> initialize() {
  vector_class<plugin> Plugins;

  if (!useBackend(SYCL_BE_PI_OPENCL) && !useBackend(SYCL_BE_PI_CUDA)) {
    die("Unknown SYCL_BE");
  }

  bool EnableTrace = (std::getenv("SYCL_PI_TRACE") != nullptr);

  vector_class<std::string> PluginNames;
  findPlugins(PluginNames);

  if (PluginNames.empty() && EnableTrace)
    std::cerr << "No Plugins Found." << std::endl;

  PiPlugin PluginInformation; // TODO: include.
  for (unsigned int I = 0; I < PluginNames.size(); I++) {
    void *Library = loadPlugin(PluginNames[I]);
    if (!Library && EnableTrace) {
      std::cerr << "Check if plugin is present. Failed to load plugin: "
                << PluginNames[I] << std::endl;
    }

    if (!bindPlugin(Library, &PluginInformation) && EnableTrace) {
      std::cerr << "Failed to bind PI APIs to the plugin: " << PluginNames[I]
                << std::endl;
    }
    if (useBackend(SYCL_BE_PI_OPENCL) &&
        PluginNames[I].find("opencl") != std::string::npos) {
      // Use the OpenCL plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(PluginInformation);
    }
    if (useBackend(SYCL_BE_PI_CUDA) &&
        PluginNames[I].find("cuda") != std::string::npos) {
      // Use the CUDA plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(PluginInformation);
    }
    Plugins.push_back(plugin(PluginInformation));
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && !XPTIInitDone))
    return Plugins;
  // Not sure this is the best place to initialize the framework; SYCL runtime
  // team needs to advise on the right place, until then we piggy-back on the
  // initialization of the PI layer.

  // Initialize the global events just once, in the case pi::initialize() is
  // called multiple times
  XPTIInitDone = true;
  // Registers a new stream for 'sycl' and any plugin that wants to listen to
  // this stream will register itself using this string or stream ID for this
  // string.
  uint8_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  //  Let all tool plugins know that a stream by the name of 'sycl' has been
  //  initialized and will be generating the trace stream.
  //
  //                                           +--- Minor version #
  //            Major version # ------+        |   Version string
  //                                  |        |       |
  //                                  v        v       v
  xptiInitialize(SYCL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  // Create a tracepoint to indicate the graph creation
  xpti::payload_t GraphPayload("application_graph");
  uint64_t GraphInstanceNo;
  GSYCLGraphEvent =
      xptiMakeEvent("application_graph", &GraphPayload, xpti::trace_graph_event,
                    xpti_at::active, &GraphInstanceNo);
  if (GSYCLGraphEvent) {
    // The graph event is a global event and will be used as the parent for
    // all nodes (command groups)
    xptiNotifySubscribers(StreamID, xpti::trace_graph_create, nullptr,
                          GSYCLGraphEvent, GraphInstanceNo, nullptr);
  }

  xpti::payload_t PIPayload("Plugin Interface Layer");
  uint64_t PiInstanceNo;
  GPICallEvent =
      xptiMakeEvent("PI Layer", &PIPayload, xpti::trace_algorithm_event,
                    xpti_at::active, &PiInstanceNo);
#endif

  return Plugins;
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

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
