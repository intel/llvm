// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_BE=PI_CUDA %GPU_RUN_PLACEHOLDER %t.out

//==--------info_ocl_version-cuda.cpp - SYCL objects get_info() test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>
#include <regex>
#include <string>

using namespace cl::sycl;

// This test checks that cl::sycl::info::device::version
// is returned in a form: <major_version>.<minor_version>

int main() {
  default_selector selector;
  device dev(selector.select_device());
  auto ocl_version = dev.get_info<info::device::version>();
  const std::regex oclVersionRegex("[0-9]\\.[0-9]");
  if (!std::regex_match(ocl_version, oclVersionRegex)) {
    std::cout << "Failed" << sd::endl;
    return 1;
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
