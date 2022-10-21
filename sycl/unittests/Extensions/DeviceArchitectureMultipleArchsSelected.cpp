//==--------------- DeviceArchitectureOneArchSelected.cpp ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

// define one of __SYCL_TARGET_INTEL_*** macro, e.g., the one for X86_64 and PVC
#define __SYCL_TARGET_INTEL_X86_64__ 1
#define __SYCL_TARGET_INTEL_GPU_PVC__ 1

#include <sycl/ext/intel/experimental/device_architecture.hpp>

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::intel::experimental;

TEST(DeviceArchitectureTest,
     DeviceArchitectureMultipleArchsSelected_Multiple_If) {
  int res = 0;
  if_architecture_is<architecture::x86_64>([&]() { res++; });
  if_architecture_is<architecture::intel_gpu_pvc>([&]() { res++; });
  ASSERT_EQ(res, 2);
}

TEST(DeviceArchitectureTest, DeviceArchitectureMultipleArchsSelected_Else_If) {
  int res = 0;
  if_architecture_is<architecture::x86_64>([&]() {
    res = 1;
  }).else_if_architecture_is<architecture::intel_gpu_pvc>([&]() { res = 2; });
  ASSERT_EQ(res, 2);
}
