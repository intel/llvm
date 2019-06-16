//==---------- pi.cpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl/detail/pi.hpp>
#include <cstdarg>
#include <iostream>
#include <map>

namespace cl {
namespace sycl {
namespace detail {

// For selection of SYCL RT back-end, now manually through the "SYCL_BE"
// environment variable.
//
enum pi_backend {
  SYCL_BE_PI_OPENCL,
  SYCL_BE_PI_OTHER
};

// Check for manually selected BE at run-time.
bool pi_use_backend(pi_backend be) {
  static const pi_backend use =
    std::map<std::string, pi_backend>{
      { "PI_OPENCL", SYCL_BE_PI_OPENCL },
      { "PI_OTHER",  SYCL_BE_PI_OTHER }
      // Any other value would yield 0 -> PI_OPENCL (current default)
    }[std::getenv("SYCL_BE")];
  return be == use;
}

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void pi_die(const char *message) {
  std::cerr << "pi_die: " << message << std::endl;
  std::terminate();
}

void pi_assert(bool condition, const char *message) {
  if (!condition)
    pi_die(message);
}

// TODO: implement a more mature and controllable tracing of PI calls.
void pi_trace(const char *format, ...) {
  static bool do_trace = std::getenv("SYCL_BE_TRACE");
  if (!do_trace)
    return;

  va_list args;
  va_start(args, format);
  vprintf(format, args);
}

extern "C" {
// TODO: change this pseudo-dispatch to plugins (ICD-like?)
// Currently this is using the low-level "ifunc" machinery to
// re-direct (with no overhead) the PI call to the underlying
// PI plugin requested by SYCL_BE environment variable (today
// only OpenCL, other would just die).
//
void __resolve_die() {
  pi_die("Unknown SYCL_BE");
}

#define PI_DISPATCH(api)                                  \
decltype(api) ocl_##api;                                  \
static void *__resolve_##api(void) {                      \
  return (pi_use_backend(SYCL_BE_PI_OPENCL) ?             \
    (void*)ocl_##api : (void*)__resolve_die);             \
}                                                         \
decltype(api) api __attribute__((ifunc ("__resolve_" #api)));

// Platform
PI_DISPATCH(piPlatformsGet)
PI_DISPATCH(piPlatformGetInfo)
// Device
PI_DISPATCH(piDevicesGet)
PI_DISPATCH(piDeviceRetain)
PI_DISPATCH(piDeviceRelease)
PI_DISPATCH(piDeviceGetInfo)
PI_DISPATCH(piDevicePartition)
// IR
PI_DISPATCH(piextDeviceSelectBinary)

} // extern "C"

} // namespace detail
} // namespace sycl
} // namespace cl
