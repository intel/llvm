// RUN: %{build} -o %t.out
// RUN: env SYCL_DEVICE_FILTER='*' %{run-unfiltered-devices} %t.out &> %t.log
// RUN: FileCheck %s < %t.log
//
// CHECK:      WARNING: The enviroment variable SYCL_DEVICE_FILTER is deprecated.
// CHECK-SAME: Please use ONEAPI_DEVICE_SELECTOR instead.
// CHECK-NEXT: For more details, please refer to:
// CHECK-NEXT: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector

//==---- deprecated_sycl_device_filter.cpp - SYCL 2020 deprecation test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------------------===//

// This test is to check if a warning message is displayed when using the
// enviroment variable SYCL_DEVICE_FILTER
// TODO: Remove test when SYCL_DEVICE_FILTER is removed
#include <sycl/sycl.hpp>

int main() {
  using namespace sycl;
  platform p{};
  auto devices = p.get_devices();
  std::cout << "Passed test\n";
  return 0;
}
