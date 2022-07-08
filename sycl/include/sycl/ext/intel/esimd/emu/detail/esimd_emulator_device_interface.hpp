//==----- esimd_emulator_device_interface.hpp - DPC++ Explicit SIMD API ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// @cond ESIMD_EMU

/// \file esimd_emulator_device_interface.hpp
/// Declarations for ESIMD_EMULATOR-device specific definitions.
/// ESIMD intrinsic and LibCM functionalities required by intrinsic defined
///
/// This interface is for ESIMD intrinsic emulation implementations
/// such as slm_access to access ESIMD_EMULATOR specific-support therefore
/// it has to be defined and shared as include directory
///
/// \ingroup sycl_pi_esimd_emulator

#pragma once

#include <CL/sycl/detail/pi.hpp>

// cstdint-type fields such as 'uint32_t' are to be used in funtion
// pointer table file ('esimd_emulator_functions_v1.h') included in 'struct
// ESIMDDeviceInterface' definition.
#include <cstdint>
#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/// This is the device interface version required (and used) by this
/// implementation of the ESIMD CPU emulator.
#define ESIMD_DEVICE_INTERFACE_VERSION 1

// 'ESIMDDeviceInterface' structure defines interface for ESIMD CPU
// emulation (ESIMD_EMULATOR) to access LibCM CPU emulation functionalities
// from kernel application under emulation.

// Header files included in the structure contains only function
// pointers to access CM functionalities. Only new function can be
// added - reordering, changing, or removing existing function pointer
// is not allowed.

// Whenever a new function(s) is added to this interface, a new header
// file must be added following naming convention that contains
// version number such as 'v1' from 'ESIMD_DEVICE_INTERFACE_VERSION'.
struct ESIMDDeviceInterface {
  uintptr_t version;
  void *reserved;

  ESIMDDeviceInterface();
#include "esimd_emulator_functions_v1.h"
};

// Denotes the data version used by the implementation.
// Increment whenever the 'data' field interpretation within PluginOpaqueData is
// changed.
#define ESIMD_EMULATOR_PLUGIN_OPAQUE_DATA_VERSION 0
/// This structure denotes a ESIMD EMU plugin-specific data returned via the
/// piextPluginGetOpaqueData PI call. Depending on the \c version field, the
/// second \c data field can be interpreted differently.
struct ESIMDEmuPluginOpaqueData {
  uintptr_t version;
  void *data;
};

__SYCL_EXPORT ESIMDDeviceInterface *getESIMDDeviceInterface();

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

/// @endcond ESIMD_EMU
