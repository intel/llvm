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

namespace cl {
namespace sycl {
namespace detail {

class pi {
public:
  using pi_result             = ::pi_result;
  using pi_platform           = ::pi_platform;
  using pi_device             = ::pi_device;
  using pi_device_type        = ::pi_device_type;
  using pi_device_binary_type = ::pi_device_binary_type;
  using pi_device_info        = ::pi_device_info;
  using pi_program            = ::pi_program;

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
  // IR
  _PI_API(piextDeviceSelectBinary)

  #undef _PI_API
};

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] void pi_die(const char *message);
void pi_assert(bool condition, const char *message = 0);

#define _PI_STRINGIZE(x) _PI_STRINGIZE2(x)
#define _PI_STRINGIZE2(x) #x
#define PI_ASSERT(cond, msg) \
  pi_assert(condition, "assert @ " __FILE__ ":" _PI_STRINGIZE(__LINE__) msg);

// This does the call, the trace and the check for no errors.
// TODO: remove dependency on CHECK_OCL_CODE.
// TODO: implement a more mature and controllable tracing of PI calls.
void pi_trace(const char *format, ...);
#define PI_CALL(pi_call) {                  \
  pi_trace("PI ---> %s\n", #pi_call);       \
  auto __result = (pi_call);                \
  pi_trace("PI <--- %d\n", __result);       \
  CHECK_OCL_CODE(__result);                 \
}

// Want all the needed casts be explicit, do not define conversion operators.
template<class To, class From>
To pi_cast(From value) {
  // TODO: see if more sanity checks are possible.
  pi_assert(sizeof(From) == sizeof(To));
  return (To)(value);
}

} // namespace detail
} // namespace sycl
} // namespace cl

