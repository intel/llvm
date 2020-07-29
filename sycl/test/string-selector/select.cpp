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

using namespace cl::sycl;
using namespace cl::sycl::ext::oneapi;

int main() {
  queue q1(string_selector("platform=Intel"));
  std::cout << q1.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q1.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q1.get_device().is_host()) {
    assert(q1.get_device().get_platform().get_info<info::platform::name>().find(
               "Intel") != std::string::npos &&
           "Intel platform not found!");
  }

  queue q2(string_selector("type=cpu"));
  std::cout << q2.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q2.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q2.get_device().is_host()) {
    assert(q2.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q3(string_selector("type=gpu"));
  std::cout << q3.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q3.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q3.get_device().is_host()) {
    assert(q3.get_device().is_gpu() && "Device is not GPU!");
  }

  queue q4(string_selector("type=cpu,gpu"));
  std::cout << q4.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q4.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q4.get_device().is_host()) {
    assert((q4.get_device().is_gpu() || q4.get_device().is_cpu()) &&
           "Device is not GPU or CPU!");
  }

  queue q5(string_selector("platform=OpenCL;type=cpu"));
  std::cout << q5.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q5.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q5.get_device().is_host()) {
    assert(q5.get_device().get_platform().get_info<info::platform::name>().find(
               "OpenCL") != std::string::npos &&
           "OpenCL platform not found!");
    assert(q5.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q6(string_selector("platform=OpenCL;type=cpu;"));
  std::cout << q6.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q6.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q6.get_device().is_host()) {
    assert(q6.get_device().get_platform().get_info<info::platform::name>().find(
               "OpenCL") != std::string::npos &&
           "OpenCL platform not found!");
    assert(q6.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q7(string_selector(";platform=OpenCL;type=cpu;"));
  std::cout << q7.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q7.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q7.get_device().is_host()) {
    assert(q7.get_device().get_platform().get_info<info::platform::name>().find(
               "OpenCL") != std::string::npos &&
           "OpenCL platform not found!");
    assert(q7.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q8(string_selector(";    platform  = OpenCL ; type=       cpu,;"));
  std::cout << q8.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q8.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;
  if (!q8.get_device().is_host()) {
    assert(q8.get_device().get_platform().get_info<info::platform::name>().find(
               "OpenCL") != std::string::npos &&
           "OpenCL platform not found!");
    assert(q8.get_device().is_cpu() && "Device is not CPU!");
  }

  queue q9(string_selector(";   ,,,, ;                       "));
  std::cout << q9.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q9.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;

  queue q10(string_selector("     "));
  std::cout << q10.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q10.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;

  queue q11(string_selector(""));
  std::cout << q11.get_device().get_info<info::device::name>() << std::endl;
  std::cout << q11.get_device().get_platform().get_info<info::platform::name>()
            << std::endl;

  try {
    queue q12(string_selector("plat=Intel"));
  } catch (runtime_error e) {
    std::cout << "TEST PASS: " << e.what() << std::endl;
  }

  try {
    queue q13(string_selector("plat_type=Foo"));
  } catch (runtime_error e) {
    std::cout << "TEST PASS: " << e.what() << std::endl;
  }

  return 0;
}
