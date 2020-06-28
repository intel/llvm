//=- SYCLRTShared.h - Shared definition between llvm tools and SYCL runtime -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_SYCLRTSHARED_H
#define LLVM_SUPPORT_SYCLRTSHARED_H
#include <cstdint>
/// Device binary image property set names recognized by the SYCL runtime.
/// Name must be consistent with
/// PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS defined in
/// PropertySetIO.h
#define PI_PROPERTY_SET_SPEC_CONST_MAP "SYCL/specialization constants"
#define PI_PROPERTY_SET_DEVICELIB_REQ_MASK "SYCL/devicelib req mask"
namespace llvm {
namespace util {
namespace sycl {
// Specific property category names used by tools.
static constexpr char SYCL_SPECIALIZATION_CONSTANTS[] =
    "SYCL/specialization constants";
static constexpr char SYCL_DEVICELIB_REQ_MASK[] = "SYCL/devicelib req mask";
enum class DeviceLibExt : std::uint32_t {
  cl_intel_devicelib_assert,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64
};
} // namespace sycl
} // namespace util
} // namespace llvm
#endif
