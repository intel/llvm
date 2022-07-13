// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

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

#include <sycl/sycl.hpp>

#include <iostream>
#include <regex>
#include <string>

using namespace cl::sycl;

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

int main(int argc, char **argv) {
  // Must be enabled at the beginning of the application
  // to obtain the PCI address
  setenv("SYCL_ENABLE_PCI", "1", 0);

  // Expected format is "{domain}:{bus}:{device}.{function} where:
  // * {domain} is a 4 character hexadecimal value.
  // * {bus} is a 2 character hexadecimal value.
  // * {device} is a 2 character hexadecimal value.
  // * {function} is a 1 character hexadecimal value.
  const std::regex ExpectedBDFFormat{
      "^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}.[0-9a-fA-F]$"};

  for (const auto &plt : platform::get_platforms()) {
    if (plt.has(aspect::host))
      continue;
    for (const auto &dev : plt.get_devices()) {
      if (!dev.has(aspect::ext_intel_pci_address))
        continue;

      std::string PCIAddress =
          dev.get_info<info::device::ext_intel_pci_address>();
      std::cout << "PCI address = " << PCIAddress << std::endl;
      assert(std::regex_match(PCIAddress, ExpectedBDFFormat));
    }
  }
  std::cout << "Passed!" << std::endl;
  return 0;
}
