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

#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi_opencl.hpp> // TODO: remove when switched to PI

namespace cl {
namespace sycl {
namespace detail {

namespace pi {

  using PiResult              = ::pi_result;
  using PiPlatform            = ::pi_platform;
  using PiDevice              = ::pi_device;
  using PiDeviceType          = ::pi_device_type;
  using PiDeviceInfo          = ::pi_device_info;
  using PiDeviceBinaryType    = ::pi_device_binary_type;
  using PiContext             = ::pi_context;
  using PiProgram             = ::pi_program;
  using PiKernel              = ::pi_kernel;
  using PiQueue               = ::pi_queue;
  using PiMem                 = ::pi_mem;
  using PiMemFlags            = ::pi_mem_flags;
  using PiEvent               = ::pi_event;
  using PiSampler             = ::pi_sampler;

  // Convinience macro to have things look compact.
  #define _PI_API(pi_api) \
    static constexpr decltype(::pi_api) * pi_api = &::pi_api;

  // Platform
  _PI_API(piPlatformsGet)
  _PI_API(piPlatformGetInfo)
  // Device
  _PI_API(piDevicesGet)
  _PI_API(piDeviceGetInfo)
  _PI_API(piDevicePartition)
  _PI_API(piDeviceRetain)
  _PI_API(piDeviceRelease)
  _PI_API(piextDeviceSelectBinary)
    // Context
  _PI_API(piContextCreate)
  _PI_API(piContextGetInfo)
  _PI_API(piContextRetain)
  _PI_API(piContextRelease)
  // Queue
  _PI_API(piQueueCreate)
  _PI_API(piQueueGetInfo)
  _PI_API(piQueueFinish)
  _PI_API(piQueueRetain)
  _PI_API(piQueueRelease)
  // Memory
  _PI_API(piMemCreate)
  _PI_API(piMemGetInfo)
  _PI_API(piMemRetain)
  _PI_API(piMemRelease)
  // Program
  _PI_API(piProgramCreate)
  _PI_API(piclProgramCreateWithSource)
  _PI_API(piclProgramCreateWithBinary)
  _PI_API(piProgramGetInfo)
  _PI_API(piProgramCompile)
  _PI_API(piProgramBuild)
  _PI_API(piProgramLink)
  _PI_API(piProgramGetBuildInfo)
  _PI_API(piProgramRetain)
  _PI_API(piProgramRelease)
  // Kernel
  _PI_API(piKernelCreate)
  _PI_API(piKernelSetArg)
  _PI_API(piKernelGetInfo)
  _PI_API(piKernelGetGroupInfo)
  _PI_API(piKernelGetSubGroupInfo)
  _PI_API(piKernelRetain)
  _PI_API(piKernelRelease)
  // Event
  _PI_API(piEventCreate)
  _PI_API(piEventGetInfo)
  _PI_API(piEventGetProfilingInfo)
  _PI_API(piEventsWait)
  _PI_API(piEventSetCallback)
  _PI_API(piEventSetStatus)
  _PI_API(piEventRetain)
  _PI_API(piEventRelease)
  // Sampler
  _PI_API(piSamplerCreate)
  _PI_API(piSamplerGetInfo)
  _PI_API(piSamplerRetain)
  _PI_API(piSamplerRelease)
  // Queue commands
  _PI_API(piEnqueueKernelLaunch)
  _PI_API(piEnqueueEventsWait)
  _PI_API(piEnqueueMemRead)
  _PI_API(piEnqueueMemReadRect)
  _PI_API(piEnqueueMemWrite)
  _PI_API(piEnqueueMemWriteRect)
  _PI_API(piEnqueueMemCopy)
  _PI_API(piEnqueueMemCopyRect)
  _PI_API(piEnqueueMemFill)
  _PI_API(piEnqueueMemMap)
  _PI_API(piEnqueueMemUnmap)

  #undef _PI_API
} // namespace pi

// Select underlying runtime interface in compile-time (OpenCL or PI).
// As such only one path (OpenCL today) is being regularily tested.
// TODO: change to
// namespace RT = cl::sycl::detail::pi;
namespace RT = cl::sycl::detail::pi_opencl;

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] void piDie(const char *Message);
void piAssert(bool Condition, const char *Message = nullptr);

#define PI_ASSERT(cond, msg) \
  piAssert((cond), "assert @ " __FILE__ ":" STRINGIFY_LINE(__LINE__) msg);

// The PiCall helper structure facilitates performing a call to PI.
// It holds utilities to do the tracing and to check the returned result.
// TODO: implement a more mature and controllable tracing of PI calls.
class PiCall {
  RT::PiResult  m_Result;
  static bool   m_TraceEnabled;

public:
  explicit PiCall(const char *Trace = nullptr);
  ~PiCall();
  RT::PiResult get(RT::PiResult Result);
  template<typename Exception>
  void check(RT::PiResult Result);
};

// This does the call, the trace and the check for no errors.
#define PI_CALL(pi) \
    detail::PiCall(#pi).check<cl::sycl::runtime_error>( \
        detail::pi_cast<detail::RT::PiResult>(pi))

// This does the trace, the call, and returns the result
#define PI_CALL_RESULT(pi) \
    detail::PiCall(#pi).get(detail::pi_cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws
#define PI_CHECK(pi) \
    detail::PiCall().check<cl::sycl::runtime_error>( \
        detail::pi_cast<detail::RT::PiResult>(pi))

// This does the check for no errors and possibly throws x
#define PI_CHECK_THROW(pi, x) \
    detail::PiCall().check<x>( \
        detail::pi_cast<detail::RT::PiResult>(pi))

// Want all the needed casts be explicit, do not define conversion operators.
template<class To, class From>
To pi_cast(From value) {
  // TODO: see if more sanity checks are possible.
  PI_ASSERT(sizeof(From) == sizeof(To), "pi_cast failed size check");
  return (To)(value);
}

} // namespace detail
} // namespace sycl
} // namespace cl
