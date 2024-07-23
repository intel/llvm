//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "MockKernelInfo.hpp"
#include "PiImage.hpp"

template <size_t KernelSize = 1> class TestKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <size_t KernelSize>
struct KernelInfo<TestKernel<KernelSize>>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernel"; }
  static constexpr int64_t getKernelSize() { return KernelSize; }
  static constexpr const char *getFileName() { return "TestKernel.hpp"; }
  static constexpr const char *getFunctionName() {
    return "TestKernelFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 13; }
  static constexpr unsigned getColumnNumber() { return 8; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage Img =
    sycl::unittest::generateDefaultImage({"TestKernel"});
static sycl::unittest::PiImageArray<1> ImgArray{&Img};
