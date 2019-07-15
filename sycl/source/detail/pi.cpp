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
#include <iostream>
#include <map>

namespace cl {
namespace sycl {
namespace detail {

// Check for manually selected BE at run-time.
bool piUseBackend(PiBackend Backend) {
  static const char *GetEnv = std::getenv("SYCL_BE");
  static const PiBackend Use =
    std::map<std::string, PiBackend>{
      { "PI_OPENCL", SYCL_BE_PI_OPENCL },
      { "PI_OTHER",  SYCL_BE_PI_OTHER }
      // Any other value would yield PI_OPENCL (current default)
    }[ GetEnv ? GetEnv : "PI_OPENCL"];
  return Backend == Use;
}

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void piDie(const char *Message) {
  std::cerr << "pi_die: " << Message << std::endl;
  std::terminate();
}

void piAssert(bool Condition, const char *Message) {
  if (!Condition)
    piDie(Message);
}

bool PiCall::m_TraceEnabled = (std::getenv("SYCL_PI_TRACE") != nullptr);

// Emits trace before the start of PI call
PiCall::PiCall(const char *Trace) {
  if (m_TraceEnabled && Trace) {
    std::cerr << "PI ---> " << Trace << std::endl;
  }
}
// Emits trace after the end of PI call
PiCall::~PiCall() {
  if (m_TraceEnabled) {
    std::cerr << "PI <--- " << m_Result << std::endl;
  }
}
// Records and returns the result of PI call
RT::PiResult PiCall::get(RT::PiResult Result) {
  m_Result = Result;
  return Result;
}
template<typename Exception>
void PiCall::check(RT::PiResult Result) {
  m_Result = Result;
  // TODO: remove dependency on CHECK_OCL_CODE_THROW.
  CHECK_OCL_CODE_THROW(Result, Exception);
}

template void PiCall::check<cl::sycl::runtime_error>(RT::PiResult);
template void PiCall::check<cl::sycl::compile_program_error>(RT::PiResult);

extern "C" {
// TODO: change this pseudo-dispatch to plugins (ICD-like?)
// Currently this is using the low-level "ifunc" machinery to
// re-direct (with no overhead) the PI call to the underlying
// PI plugin requested by SYCL_BE environment variable (today
// only OpenCL, other would just die).
//
void __resolve_die() {
  piDie("Unknown SYCL_BE");
}

#define _PI_DISPATCH(api)                                 \
decltype(api) ocl_##api;                                  \
static void *__resolve_##api(void) {                      \
  return (piUseBackend(SYCL_BE_PI_OPENCL) ?               \
    (void*)ocl_##api : (void*)__resolve_die);             \
}                                                         \
decltype(api) api __attribute__((ifunc ("__resolve_" #api)));

// Platform
_PI_DISPATCH(piPlatformsGet)
_PI_DISPATCH(piPlatformGetInfo)
// Device
_PI_DISPATCH(piDevicesGet)
_PI_DISPATCH(piDeviceGetInfo)
_PI_DISPATCH(piDevicePartition)
_PI_DISPATCH(piDeviceRetain)
_PI_DISPATCH(piDeviceRelease)
_PI_DISPATCH(piextDeviceSelectBinary)
  // Context
_PI_DISPATCH(piContextCreate)
_PI_DISPATCH(piContextGetInfo)
_PI_DISPATCH(piContextRetain)
_PI_DISPATCH(piContextRelease)
// Queue
_PI_DISPATCH(piQueueCreate)
_PI_DISPATCH(piQueueGetInfo)
_PI_DISPATCH(piQueueFinish)
_PI_DISPATCH(piQueueRetain)
_PI_DISPATCH(piQueueRelease)
// Memory
_PI_DISPATCH(piMemCreate)
_PI_DISPATCH(piMemGetInfo)
_PI_DISPATCH(piMemRetain)
_PI_DISPATCH(piMemRelease)
// Program
_PI_DISPATCH(piProgramCreate)
_PI_DISPATCH(piclProgramCreateWithSource)
_PI_DISPATCH(piclProgramCreateWithBinary)
_PI_DISPATCH(piProgramGetInfo)
_PI_DISPATCH(piProgramCompile)
_PI_DISPATCH(piProgramBuild)
_PI_DISPATCH(piProgramLink)
_PI_DISPATCH(piProgramGetBuildInfo)
_PI_DISPATCH(piProgramRetain)
_PI_DISPATCH(piProgramRelease)
// Kernel
_PI_DISPATCH(piKernelCreate)
_PI_DISPATCH(piKernelSetArg)
_PI_DISPATCH(piKernelGetInfo)
_PI_DISPATCH(piKernelGetGroupInfo)
_PI_DISPATCH(piKernelGetSubGroupInfo)
_PI_DISPATCH(piKernelRetain)
_PI_DISPATCH(piKernelRelease)
// Event
_PI_DISPATCH(piEventCreate)
_PI_DISPATCH(piEventGetInfo)
_PI_DISPATCH(piEventGetProfilingInfo)
_PI_DISPATCH(piEventsWait)
_PI_DISPATCH(piEventSetCallback)
_PI_DISPATCH(piEventSetStatus)
_PI_DISPATCH(piEventRetain)
_PI_DISPATCH(piEventRelease)
// Sampler
_PI_DISPATCH(piSamplerCreate)
_PI_DISPATCH(piSamplerGetInfo)
_PI_DISPATCH(piSamplerRetain)
_PI_DISPATCH(piSamplerRelease)
// Queue commands
_PI_DISPATCH(piEnqueueKernelLaunch)
_PI_DISPATCH(piEnqueueEventsWait)
_PI_DISPATCH(piEnqueueMemRead)
_PI_DISPATCH(piEnqueueMemReadRect)
_PI_DISPATCH(piEnqueueMemWrite)
_PI_DISPATCH(piEnqueueMemWriteRect)
_PI_DISPATCH(piEnqueueMemCopy)
_PI_DISPATCH(piEnqueueMemCopyRect)
_PI_DISPATCH(piEnqueueMemFill)
_PI_DISPATCH(piEnqueueMemMap)
_PI_DISPATCH(piEnqueueMemUnmap)

} // extern "C"

} // namespace detail
} // namespace sycl
} // namespace cl
