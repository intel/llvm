//==---------- pi.hpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.hpp
/// C++ wrapper of extern "C" PI interfaces
///
/// \ingroup sycl_pi

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.h>
#include <sstream>

#include <cassert>
#include <string>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

enum class PiApiKind {
#define _PI_API(api) api,
#include <CL/sycl/detail/pi.def>
};
class plugin;
namespace pi {

#ifdef SYCL_RT_OS_WINDOWS
#define OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define CUDA_PLUGIN_NAME "pi_cuda.dll"
#else
#define OPENCL_PLUGIN_NAME "libpi_opencl.so"
#define CUDA_PLUGIN_NAME "libpi_cuda.so"
#endif

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] void die(const char *Message);

void assertion(bool Condition, const char *Message = nullptr);

template <typename T>
void handleUnknownParamName(const char *functionName, T parameter) {
  std::stringstream stream;
  stream << "Unknown parameter " << parameter << " passed to " << functionName
         << "\n";
  auto str = stream.str();
  auto msg = str.c_str();
  die(msg);
}

// This macro is used to report invalid enumerators being passed to PI API
// GetInfo functions. It will print the name of the function that invoked it
// and the value of the unknown enumerator.
#define PI_HANDLE_UNKNOWN_PARAM_NAME(parameter)                                \
  { cl::sycl::detail::pi::handleUnknownParamName(__func__, parameter); }

using PiPlugin = ::pi_plugin;
using PiResult = ::pi_result;
using PiPlatform = ::pi_platform;
using PiDevice = ::pi_device;
using PiDeviceType = ::pi_device_type;
using PiDeviceInfo = ::pi_device_info;
using PiDeviceBinaryType = ::pi_device_binary_type;
using PiContext = ::pi_context;
using PiProgram = ::pi_program;
using PiKernel = ::pi_kernel;
using PiQueue = ::pi_queue;
using PiQueueProperties = ::pi_queue_properties;
using PiMem = ::pi_mem;
using PiMemFlags = ::pi_mem_flags;
using PiEvent = ::pi_event;
using PiSampler = ::pi_sampler;
using PiSamplerInfo = ::pi_sampler_info;
using PiSamplerProperties = ::pi_sampler_properties;
using PiSamplerAddressingMode = ::pi_sampler_addressing_mode;
using PiSamplerFilterMode = ::pi_sampler_filter_mode;
using PiMemImageFormat = ::pi_image_format;
using PiMemImageDesc = ::pi_image_desc;
using PiMemImageInfo = ::pi_image_info;
using PiMemObjectType = ::pi_mem_type;
using PiMemImageChannelOrder = ::pi_image_channel_order;
using PiMemImageChannelType = ::pi_image_channel_type;

// Function to load the shared library
// Implementation is OS dependent.
void *loadOsLibrary(const std::string &Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// For selection of SYCL RT back-end, now manually through the "SYCL_BE"
// environment variable.
enum Backend { SYCL_BE_PI_OPENCL, SYCL_BE_PI_CUDA, SYCL_BE_PI_OTHER };

// Check for manually selected BE at run-time.
bool useBackend(Backend Backend);

// Get a string representing a _pi_platform_info enum
std::string platformInfoToString(pi_platform_info info);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Holds the PluginInformation for the plugin that is bound.
// Currently a global variable is used to store OpenCL plugin information to be
// used with SYCL Interoperability Constructors.
extern std::shared_ptr<plugin> GlobalPlugin;

// Performs PI one-time initialization.
vector_class<plugin> initialize();

// Utility Functions to get Function Name for a PI Api.
template <PiApiKind PiApiOffset> struct PiFuncInfo {};

#define _PI_API(api)                                                           \
  template <> struct PiFuncInfo<PiApiKind::api> {                              \
    inline std::string getFuncName() { return #api; }                          \
    inline decltype(&::api) getFuncPtr(PiPlugin MPlugin) {                     \
      return MPlugin.PiFunctionTable.api;                                      \
    }                                                                          \
  };
#include <CL/sycl/detail/pi.def>

// Helper utilities for PI Tracing
// The run-time tracing of PI calls.
// Print functions used by Trace class.
template <typename T> inline void print(T val) {
  std::cout << "<unknown> : " << val << std::endl;
}

template <> inline void print<>(PiPlatform val) {
  std::cout << "pi_platform : " << val << std::endl;
}

template <> inline void print<>(PiResult val) {
  std::cout << "pi_result : ";
  if (val == PI_SUCCESS)
    std::cout << "PI_SUCCESS" << std::endl;
  else
    std::cout << val << std::endl;
}

// cout does not resolve a nullptr.
template <> inline void print<>(std::nullptr_t val) { print<void *>(val); }

inline void printArgs(void) {}
template <typename Arg0, typename... Args>
void printArgs(Arg0 arg0, Args... args) {
  std::cout << "       ";
  print(arg0);
  printArgs(std::forward<Args>(args)...);
}
} // namespace pi

namespace RT = cl::sycl::detail::pi;

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> To pi::cast(From value) {
  // TODO: see if more sanity checks are possible.
  RT::assertion((sizeof(From) == sizeof(To)), "assert: cast failed size check");
  return (To)(value);
}

} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = cl::sycl::detail::pi;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
