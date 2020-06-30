//=- SYCLRTShared.h - Shared definition between llvm tools and SYCL runtime -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Shares type definitions between llvm tools and SYCL RT library
// Definitions which introduce SYCL RT - LLVM linkage dependencies should not
// be added here
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_SYCLRTSHARED_H
#define LLVM_SUPPORT_SYCLRTSHARED_H
#include <cstdint>
#define PROP_SYCL_SPECIALIZATION_CONSTANTS "SYCL/specialization constants"
#define PROP_SYCL_DEVICELIB_REQ_MASK "SYCL/devicelib req mask"
namespace llvm {
namespace util {
namespace sycl {
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
