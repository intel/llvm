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
  static const Backend Use =
    std::map<std::string, Backend>{
      { "PI_OPENCL", SYCL_BE_PI_OPENCL },
      { "PI_OTHER",  SYCL_BE_PI_OTHER }
      // Any other value would yield PI_OPENCL (current default)
    }[ GetEnv ? GetEnv : "PI_OPENCL"];
  return TheBackend == Use;
}

// Definitions of the PI dispatch entries, they will be initialized
// at their first use with piInitialize.
#define _PI_API(api) decltype(::api) * api = nullptr;
#include <CL/sycl/detail/pi.def>

// TODO: implement real plugins (ICD-like?)
// For now this has the effect of redirecting to built-in PI OpenCL plugin.
void initialize() {
  static bool Initialized = false;
  if (Initialized) {
    return;
  }
  if (!useBackend(SYCL_BE_PI_OPENCL)) {
    die("Unknown SYCL_BE");
  }
  #define _PI_API(api)                          \
    extern const decltype(::api) * api##OclPtr; \
    api = api##OclPtr;
  #include <CL/sycl/detail/pi.def>

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

} // namespace pi
} // namespace detail
} // namespace sycl
} // namespace cl
