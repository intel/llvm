// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------- pipes_info.cpp - SYCL device pipe info test --*- C++ -*-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;
  sycl::device Device = Queue.get_device();
  sycl::platform Platform = Device.get_platform();

  // Query if the device supports kernel to kernel pipe feature
  bool IsSupported =
      Device.get_info<sycl::info::device::kernel_kernel_pipe_support>();

  // Query for platform string. We expect only Intel FPGA platforms to support
  // SYCL_ext_intel_data_flow_pipes extension.
  std::string platform_name = Platform.get_info<sycl::info::platform::name>();
  bool SupposedToBeSupported =
      (platform_name == "Intel(R) FPGA Emulation Platform for OpenCL(TM)" ||
       platform_name == "Intel(R) FPGA SDK for OpenCL(TM)")
          ? true
          : false;

  return (SupposedToBeSupported != IsSupported);
}
