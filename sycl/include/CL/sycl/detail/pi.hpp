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

#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/common.hpp>

namespace cl {
namespace sycl {
namespace detail {
namespace pi {
  // For selection of SYCL RT back-end, now manually through the "SYCL_BE"
  // environment variable.
  //
  enum PiBackend {
    SYCL_BE_PI_OPENCL,
    SYCL_BE_PI_OTHER
  };

  // Check for manually selected BE at run-time.
  bool piUseBackend(PiBackend Backend);

  using PiResult               = ::pi_result;
  using PiPlatform             = ::pi_platform;
  using PiDevice               = ::pi_device;
  using PiDeviceType           = ::pi_device_type;
  using PiDeviceInfo           = ::pi_device_info;
  using PiDeviceBinaryType     = ::pi_device_binary_type;
  using PiContext              = ::pi_context;
  using PiProgram              = ::pi_program;
  using PiKernel               = ::pi_kernel;
  using PiQueue                = ::pi_queue;
  using PiQueueProperties      = ::pi_queue_properties;
  using PiMem                  = ::pi_mem;
  using PiMemFlags             = ::pi_mem_flags;
  using PiEvent                = ::pi_event;
  using PiSampler              = ::pi_sampler;
  using PiMemImageFormat       = ::pi_image_format;
  using PiMemImageDesc         = ::pi_image_desc;
  using PiMemImageInfo         = ::pi_image_info;
  using PiMemObjectType        = ::pi_mem_type;
  using PiMemImageChannelOrder = ::pi_image_channel_order;
  using PiMemImageChannelType  = ::pi_image_channel_type;

  // Get a string representing a _pi_platform_info enum
  std::string platformInfoToString(pi_platform_info info);

  // Report error and no return (keeps compiler happy about no return statements).
  [[noreturn]] void piDie(const char *Message);
  void piAssert(bool Condition, const char *Message = nullptr);

  // Want all the needed casts be explicit, do not define conversion operators.
  template<class To, class From>
  To pi_cast(From value);

  // Forward declarations of the PI dispatch entries.
  #define _PI_API(api) __SYCL_EXPORTED extern decltype(::api) * api;
  #include <CL/sycl/detail/pi.def>

  // Performs PI one-time initialization.
  void piInitialize();

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
} // namespace pi

namespace RT = cl::sycl::detail::pi;

#define PI_ASSERT(cond, msg) \
  RT::piAssert((cond), "assert @ " __FILE__ ":" STRINGIFY_LINE(__LINE__) msg);

// This does the call, the trace and the check for no errors.
#define PI_CALL(pi)                                   \
    RT::piInitialize(),                               \
    RT::PiCall(#pi).check<cl::sycl::runtime_error>(   \
        RT::pi_cast<detail::RT::PiResult>(pi))

// This does the trace, the call, and returns the result
#define PI_CALL_RESULT(pi) \
    RT::PiCall(#pi).get(detail::RT::pi_cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws
#define PI_CHECK(pi) \
    RT::PiCall().check<cl::sycl::runtime_error>( \
       RT::pi_cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws x
#define PI_CHECK_THROW(pi, x) \
    RT::PiCall().check<x>( \
        RT::pi_cast<detail::RT::PiResult>(pi))

// Want all the needed casts be explicit, do not define conversion operators.
template<class To, class From>
To pi::pi_cast(From value) {
  // TODO: see if more sanity checks are possible.
  PI_ASSERT(sizeof(From) == sizeof(To), "pi_cast failed size check");
  return (To)(value);
}

} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = cl::sycl::detail::pi;

} // namespace sycl
} // namespace cl
