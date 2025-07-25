//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "MockDeviceImage.hpp"
#include "MockKernelInfo.hpp"

class TestKernel;
class TestKernelWithAcc;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernel"; }
  static constexpr int64_t getKernelSize() { return 1; }
  static constexpr const char *getFileName() { return "TestKernel.hpp"; }
  static constexpr const char *getFunctionName() {
    return "TestKernelFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 14; }
  static constexpr unsigned getColumnNumber() { return 8; }
};

template <>
struct KernelInfo<TestKernelWithAcc> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelWithAcc"; }
  static constexpr int64_t getKernelSize() {
    return sizeof(sycl::accessor<int, 0, sycl::access::mode::read_write,
                                 sycl::target::device>);
  }
  static constexpr const char *getFileName() { return "TestKernel.hpp"; }
  static constexpr const char *getFunctionName() {
    return "TestKernelWithAccFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 15; }
  static constexpr unsigned getColumnNumber() { return 8; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Imgs[] = {
    sycl::unittest::generateDefaultImage({"TestKernel"}),
    sycl::unittest::generateDefaultImage({"TestKernelWithAcc"})};
static sycl::unittest::MockDeviceImageArray<2> ImgArray{Imgs};
