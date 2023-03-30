// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env %CPU_RUN_PLACEHOLDER %t.out
// RUN: env %GPU_RUN_PLACEHOLDER %t.out
// RUN: env %ACC_RUN_PLACEHOLDER %t.out

//==--------info_ocl_version.cpp - SYCL objects get_info() test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <regex>
#include <string>
#include <sycl/sycl.hpp>

using namespace sycl;

// This test checks that sycl::info::device::version
// is returned in a form: <major_version>.<minor_version>

int main() {
  default_selector selector;
  device dev(selector.select_device());
  auto ocl_version = dev.get_info<info::device::version>();
  std::cout << ocl_version << std::endl;
  const std::regex oclVersionRegex("[0-9]\\.[0-9]");
  if (!std::regex_match(ocl_version, oclVersionRegex)) {
    std::cout << "Failed" << std::endl;
    return 1;
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
