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

namespace cl {
namespace sycl {
namespace detail {
namespace pi {
  // For selection of SYCL RT back-end, now manually through the "SYCL_BE"
  // environment variable.
  //
  enum Backend {
    SYCL_BE_PI_OPENCL,
    SYCL_BE_PI_OTHER
  };

  // Check for manually selected BE at run-time.
  bool useBackend(Backend Backend);

  using PiResult                = ::pi_result;
  using PiPlatform              = ::pi_platform;
  using PiDevice                = ::pi_device;
  using PiDeviceType            = ::pi_device_type;
  using PiDeviceInfo            = ::pi_device_info;
  using PiDeviceBinaryType      = ::pi_device_binary_type;
  using PiContext               = ::pi_context;
  using PiProgram               = ::pi_program;
  using PiKernel                = ::pi_kernel;
  using PiQueue                 = ::pi_queue;
  using PiQueueProperties       = ::pi_queue_properties;
  using PiMem                   = ::pi_mem;
  using PiMemFlags              = ::pi_mem_flags;
  using PiEvent                 = ::pi_event;
  using PiSampler               = ::pi_sampler;
  using PiSamplerInfo           = ::pi_sampler_info;
  using PiSamplerProperties     = ::pi_sampler_properties;
  using PiSamplerAddressingMode = ::pi_sampler_addressing_mode;
  using PiSamplerFilterMode     = ::pi_sampler_filter_mode;
  using PiMemImageFormat        = ::pi_image_format;
  using PiMemImageDesc          = ::pi_image_desc;
  using PiMemImageInfo          = ::pi_image_info;
  using PiMemObjectType         = ::pi_mem_type;
  using PiMemImageChannelOrder  = ::pi_image_channel_order;
  using PiMemImageChannelType   = ::pi_image_channel_type;

  // Get a string representing a _pi_platform_info enum
  std::string platformInfoToString(pi_platform_info info);

  // Report error and no return (keeps compiler happy about no return statements).
  [[noreturn]] void die(const char *Message);
  void assertion(bool Condition, const char *Message = nullptr);

  // Want all the needed casts be explicit, do not define conversion operators.
  template<class To, class From>
  To cast(From value);

  // Forward declarations of the PI dispatch entries.
#define _PI_API(api) __SYCL_EXPORTED extern decltype(::api) *(api);
#include <CL/sycl/detail/pi.def>

  // Performs PI one-time initialization.
  void initialize();

  // The PiCall helper structure facilitates performing a call to PI.
  // It holds utilities to do the tracing and to check the returned result.
  // TODO: implement a more mature and controllable tracing of PI calls.
  class PiCall {
    PiResult m_Result;
    static bool m_TraceEnabled;

  public:
    explicit PiCall(const char *Trace = nullptr);
    ~PiCall();
    PiResult get(PiResult Result);
    template<typename Exception>
    void check(PiResult Result);
  };

  // The run-time tracing of PI calls.
  // TODO: replace PiCall completely with this one (PiTrace)
  //
  template <typename T> inline
  void print(T val) {
    std::cout << "<unknown> : " << val;
  }

  template<> inline void print<> (PiPlatform val) { std::cout << "pi_platform : " << val; }
  template<> inline void print<> (PiResult val) {
    std::cout << "pi_result : ";
    if (val == PI_SUCCESS)
      std::cout << "PI_SUCCESS";
    else
      std::cout << val; 
  }
  
  inline void printArgs(void) {}
  template <typename Arg0, typename... Args>
  void printArgs(Arg0 arg0, Args... args) {
    std::cout << std::endl << "       ";
    print(arg0);
    printArgs(std::forward<Args>(args)...);
  }
  
  template <typename FnType>
  class Trace {
  private:
    FnType m_FnPtr;
    static bool m_TraceEnabled;
  public:
    Trace(FnType FnPtr, const std::string &FnName) : m_FnPtr(FnPtr) {
      if (m_TraceEnabled)
        std::cout << "---> " << FnName << "(";
    }
  
    template <typename... Args>
    typename std::result_of<FnType(Args...)>::type
    operator() (Args... args) {
      if (m_TraceEnabled)
        printArgs(args...);

      initialize();
      auto r = m_FnPtr(args...);

      if (m_TraceEnabled) {
        std::cout << ") ---> ";
        std::cout << (print(r),"") << "\n";
      }
      return r;
    }
  };

  template <typename FnType>
  bool Trace<FnType>::m_TraceEnabled = (std::getenv("SYCL_PI_TRACE") != nullptr);

} // namespace pi

namespace RT = cl::sycl::detail::pi;

#define PI_ASSERT(cond, msg) \
  RT::assertion((cond), "assert: " msg);

#define PI_TRACE(func) RT::Trace<decltype(func)>(func, #func)

// This does the call, the trace and the check for no errors.
#define PI_CALL(pi)                                   \
    RT::initialize(),                                 \
    RT::PiCall(#pi).check<cl::sycl::runtime_error>(   \
        RT::cast<detail::RT::PiResult>(pi))

// This does the trace, the call, and returns the result
#define PI_CALL_RESULT(pi) \
    RT::PiCall(#pi).get(detail::RT::cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws
#define PI_CHECK(pi) \
    RT::PiCall().check<cl::sycl::runtime_error>( \
       RT::cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws x
#define PI_CHECK_THROW(pi, x) \
    RT::PiCall().check<x>( \
        RT::cast<detail::RT::PiResult>(pi))

// Want all the needed casts be explicit, do not define conversion operators.
template<class To, class From>
To pi::cast(From value) {
  // TODO: see if more sanity checks are possible.
  PI_ASSERT(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
}

} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = cl::sycl::detail::pi;

} // namespace sycl
} // namespace cl
