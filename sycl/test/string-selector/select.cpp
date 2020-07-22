// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %t1.out

// REQUIRES: cpu, gpu, opencl

//==------------------- select.cpp - string_selector test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

int main() {
  queue q1(string_selector("plat=Intel"));
  std::cout << q1.get_device().get_info<info::device::name>() << std::endl;
  if (!q1.get_device().is_host()) {
    assert(q1.get_device().get_platform().get_info<info::platform::name>().find(
               "Intel") != std::string::npos &&
           "Intel platform not found!");
  }

  queue q2(string_selector("type=cpu"));
  std::cout << q2.get_device().get_info<info::device::name>() << std::endl;
  if (!q2.get_device().is_host()) {
    assert(q2.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q3(string_selector("type=gpu"));
  std::cout << q3.get_device().get_info<info::device::name>() << std::endl;
  if (!q3.get_device().is_host()) {
    assert(q3.get_device().is_gpu() && "Device is not GPU!");
  }

  queue q4(string_selector("type=cpu,gpu"));
  std::cout << q4.get_device().get_info<info::device::name>() << std::endl;
  if (!q4.get_device().is_host()) {
    assert((q4.get_device().is_gpu() || q4.get_device().is_cpu()) &&
           "Device is not GPU or CPU!");
  }

  queue q5(string_selector("platform=OpenCL;type=gpu"));
  std::cout << q5.get_device().get_info<info::device::name>() << std::endl;
  if (!q5.get_device().is_host()) {
    assert(q5.get_device().get_platform().get_info<info::platform::name>().find(
               "OpenCL") != std::string::npos &&
           "OpenCL platform not found!");
    assert(q5.get_device().is_gpu() && "Device is not GPU!");
  }

  return 0;
}
