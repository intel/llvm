//==--------------- DeviceArchitectureOneArchSelected.cpp ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

// define one of __SYCL_TARGET_ macro, e.g., the one for SKL
#define __SYCL_TARGET_INTEL_GPU_SKL__ 1

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::oneapi::experimental;

TEST(DeviceArchitectureTest, DeviceArchitecture_If) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_skl>([&]() { res = true; });
  ASSERT_TRUE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_If_Negative) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_pvc>([&]() { res = true; });
  ASSERT_FALSE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_Else_If) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_dg1>([]() {
  }).else_if_architecture_is<architecture::intel_gpu_skl>([&]() {
    res = true;
  });
  ASSERT_TRUE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_Otherwise) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_dg1>([]() {
  }).else_if_architecture_is<architecture::intel_gpu_pvc>([&]() {
    }).otherwise([&]() { res = true; });
  ASSERT_TRUE(res);
}
