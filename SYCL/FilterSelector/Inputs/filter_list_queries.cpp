//==--- filter_list_queries.cpp - Check available platform and devices ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <iostream>
#include <map>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main() {

  std::map<info::device_type, std::string> m = {
      {info::device_type::cpu, "cpu"},
      {info::device_type::gpu, "gpu"},
      {info::device_type::accelerator, "acc"},
      {info::device_type::host, "host"},
      {info::device_type::all, "all"}};

  for (auto &d : device::get_devices()) {
    std::cout << "Device: " << m[d.get_info<info::device::device_type>()]
              << std::endl;
  }

  return 0;
}
