//==---------- pi.hpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// C++ wrapper of extern "C" PI interfaces
//
#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.h>

#include <cassert>
#include <string>

namespace cl {
namespace sycl {
namespace detail {
namespace pi {

// Function to load the shared library
// Implementation is OS dependent.
void *loadOsLibrary(const std::string &Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// For selection of SYCL RT back-end, now manually through the "SYCL_BE"
// environment variable.
//
enum Backend { SYCL_BE_PI_OPENCL, SYCL_BE_PI_OTHER };

#ifdef SYCL_RT_OS_WINDOWS
#define PLUGIN_NAME "pi_opencl.dll"
#else
#define PLUGIN_NAME "libpi_opencl.so"
#endif

// Check for manually selected BE at run-time.
bool useBackend(Backend Backend);

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

// Get a string representing a _pi_platform_info enum
std::string platformInfoToString(pi_platform_info info);

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] void die(const char *Message);
void assertion(bool Condition, const char *Message = nullptr);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Holds the PluginInformation for the plugin that is bound.
// TODO: Move this into sycl::platform. Currenlty, we have only a single Plugin
// connection possible.
extern pi_plugin PluginInformation;

// Performs PI one-time initialization.
void initialize();

// The run-time tracing of PI calls.
// Print functions used by Trace class.
template <typename T> inline void print(T val) {
  std::cout << "<unknown> : " << val;
}

template <> inline void print<>(PiPlatform val) {
  std::cout << "pi_platform : " << val;
}

template <> inline void print<>(PiResult val) {
  std::cout << "pi_result : ";
  if (val == PI_SUCCESS)
    std::cout << "PI_SUCCESS";
  else
    std::cout << val;
}

// cout does not resolve a nullptr.
template <> inline void print<>(std::nullptr_t val) { print<void *>(val); }

inline void printArgs(void) {}
template <typename Arg0, typename... Args>
void printArgs(Arg0 arg0, Args... args) {
  std::cout << std::endl << "       ";
  print(arg0);
  printArgs(std::forward<Args>(args)...);
}

// Utility function to check return from pi calls.
// Throws if pi_result is not a PI_SUCCESS.
// TODO: Absorb this utility in Trace Class
template <typename Exception> inline void piCheckThrow(PiResult pi_result) {
  CHECK_OCL_CODE_THROW(pi_result, Exception);
}

// Utility function to check if return from pi call is
// PI_SUCCESS. If is it not, throw a cl::sycl::runtime_error.
// TODO: Absorb this utility in Trace Class
inline void piCheckResult(PiResult pi_result) {
  piCheckThrow<cl::sycl::runtime_error>(pi_result);
}

#define PI_TRACE_ENABLED (std::getenv("SYCL_PI_TRACE") != nullptr)

template <typename FnType, size_t FnOffset> class Trace {
private:
  FnType m_FnPtr;
  std::string m_FnName;

public:
  Trace();
  template <typename... Args> PiResult operator()(Args... args) {
    if (PI_TRACE_ENABLED) {
      std::cout << "---> " << m_FnName << "(";
      printArgs(args...);
    }

    PiResult r = m_FnPtr(args...);

    if (PI_TRACE_ENABLED) {
      std::cout << ") ---> ";
      std::cout << (print(r), "") << std::endl;
    }
    return r;
  }
};

template <typename FnType, size_t FnOffset>
class TraceCheck : private Trace<FnType, FnOffset> {
public:
  TraceCheck() : Trace<FnType, FnOffset>(){};

  template <typename Exception = cl::sycl::runtime_error, typename... Args>
  void operator()(Args... args) {
    PiResult Err = (Trace<FnType, FnOffset>::operator()(args...));
    piCheckThrow<Exception>(Err);
  }
};

// Explicit specialization declarations for Trace class for every FnType.
// The offsetof is used as a template argument to uniquely identify every
// api.
#define _PI_API(api)                                                           \
  template <>                                                                  \
  Trace<decltype(&::api),                                                      \
        (offsetof(_pi_plugin::FunctionPointers, api))>::Trace();

#include <CL/sycl/detail/pi.def>

} // namespace pi

namespace RT = cl::sycl::detail::pi;

#define PI_ASSERT(cond, msg) RT::assertion((cond), "assert: " msg);

// Use this macro to call the API, trace the call, check the return and throw a
// runtime_error exception.
// Usage: PI_CALL(pi)(Args);
// Note: To change the exception type, use:
// PI_CALL(pi).template operator()<compile_program_error>(__VA_ARGS__)
// Or
// auto Err = PI_CALL_NOCHECK(pi)(args);
// RT::piCheckThrow<Exception>(Err);
#define PI_CALL(pi)                                                            \
  RT::TraceCheck<decltype(&::pi),                                              \
                 (offsetof(_pi_plugin::FunctionPointers, pi))>()

// Use this macro to call the API, trace the call and return the result.
// To check the result use piCheckResult or piCheckThrow.
// Usage:
// PiResult Err = PI_CALL_NOCHECK(pi)(args);
// RT::piCheckResult(Err); <- Checks Result and throws a runtime error
// exception.
#define PI_CALL_NOCHECK(pi)                                                    \
  RT::Trace<decltype(&::pi), (offsetof(_pi_plugin::FunctionPointers, pi))>()

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> To pi::cast(From value) {
  // TODO: see if more sanity checks are possible.
  PI_ASSERT(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
}

} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = cl::sycl::detail::pi;

} // namespace sycl
} // namespace cl
