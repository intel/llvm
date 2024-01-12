//==--------------- esimd_emulator_device_interface.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file esimdcpu_device_interface.cpp
/// Definitions for ESIMD_EMULATOR-device specific definitions.
///
/// This interface is for ESIMD intrinsic emulation implementations
/// such as slm_access to access ESIMD_EMULATOR specific-support therefore
/// it has to be defined and shared as include directory
///
/// \ingroup sycl_pi_esimd_emulator

#include <sycl/detail/common.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

struct ESIMDDeviceInterface {
  uintptr_t version;
  void *reserved;
  ESIMDDeviceInterface();
};

// TODO: this function is kept only for libsycl binary backward compatibility.
// Remove it when ABI breaking changes are allowed.
__SYCL_EXPORT ESIMDDeviceInterface *getESIMDDeviceInterface() {
  return nullptr;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
