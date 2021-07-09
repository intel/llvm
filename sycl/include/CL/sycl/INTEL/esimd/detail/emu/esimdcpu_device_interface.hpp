//==----- esimdcpu_device_interface.hpp - DPC++ Explicit SIMD API ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file esimdcpu_device_interface.hpp
/// Declarations for ESIMD_CPU-device specific definitions.
/// ESIMD intrinsic and LibCM functionalities required by intrinsic defined
///
/// This interface is for ESIMD intrinsic emulation implementations
/// such as slm_access to access ESIMD_CPU specific-support therefore
/// it has to be defined and shared as include directory
///
/// \ingroup sycl_pi_esimd_cpu

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

namespace {
[[__SYCL2020_DEPRECATED(
    "include sycl/ext/intel/esimd/detail/emu/esimdcpu_device_interface.hpp "
    "instead")]] constexpr static bool HeaderDeprecated = true;
constexpr static bool TriggerHeaderDeprecationWarning = HeaderDeprecated;
} // namespace

#include <sycl/ext/intel/esimd/detail/emu/esimdcpu_device_interface.hpp>
