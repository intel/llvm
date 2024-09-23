// REQUIRES: gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- device_pci_address_bdf_format.cpp - SYCL PCI address BDF format test -==//
//
// Tests the BDF format of the PCI address reported through the corresponding
// query.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <iostream>
#include <regex>
#include <string>

using namespace sycl;

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

int main(int argc, char **argv) {
  // Expected format is "{domain}:{bus}:{device}.{function} where:
  // * {domain} is a 4 character hexadecimal value.
  // * {bus} is a 2 character hexadecimal value.
  // * {device} is a 2 character hexadecimal value.
  // * {function} is a 1 character hexadecimal value.
  const std::regex ExpectedBDFFormat{
      "^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}.[0-9a-fA-F]$"};

  for (const auto &plt : platform::get_platforms()) {
    for (const auto &dev : plt.get_devices()) {
      if (!dev.has(aspect::ext_intel_pci_address))
        continue;

      std::string PCIAddress =
          dev.get_info<ext::intel::info::device::pci_address>();
      std::cout << "PCI address = " << PCIAddress << std::endl;
      assert(std::regex_match(PCIAddress, ExpectedBDFFormat));
    }
  }
  std::cout << "Passed!" << std::endl;
  return 0;
}
