//=- SYCLRTShared.h - Shared definition between llvm tools and SYCL runtime -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_SYCLRTSHARED_H
#define LLVM_SUPPORT_SYCLRTSHARED_H
namespace llvm {
namespace util {
namespace sycl {
enum DeviceLibExt {
  cl_intel_devicelib_assert = 0,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64
};
}
} // namespace util
} // namespace llvm
#endif
