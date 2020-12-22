//===---------- pi_esimd_cpu.hpp - CM Emulation Plugin
//-------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_esimd_cpu CM Emulation Plugin
/// \ingroup sycl_pi

/// \file pi_esimd_cpu.hpp
/// Declarations for CM Emulation Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CM Emulation
///
/// \ingroup sycl_pi_esimd_cpu

#ifndef PI_ESIMD_CPU_HPP
#define PI_ESIMD_CPU_HPP

#include <CL/sycl/INTEL/esimd/esimd_libcm.hpp>
#include <CL/sycl/detail/pi.h>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include <malloc.h>

namespace cm_support {
#include <cm_rt.h>
} // namespace cm_support

template <class To, class From> To pi_cast(From Value) {
  // TODO: see if more sanity checks are possible.
  assert(sizeof(From) == sizeof(To));
  return (To)(Value);
}

template <> uint32_t pi_cast(uint64_t Value) {
  // Cast value and check that we don't lose any information.
  uint32_t CastedValue = (uint32_t)(Value);
  assert((uint64_t)CastedValue == Value);
  return CastedValue;
}

// TODO: Currently die is defined in each plugin. Probably some
// common header file with utilities should be created.
[[noreturn]] void die(const char *Message) {
  std::cerr << "die: " << Message << std::endl;
  std::terminate();
}

#include <CL/sycl/INTEL/esimd/detail/cmrt_if_defs.hpp>

#endif // PI_ESIMD_CPU_HPP
