//===------- main.cpp - Fuzz SYCL config file parsing ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  // Create (or re-create) a temporary file which will store fuzzed config and
  // write fuzzed data into it.
  {
    std::fstream file("sycl-config-temp.txt", std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to create sycl-config-temp.txt!" << std::endl;
      return 1;
    }

    try {
      file.write(reinterpret_cast<char *>(data), size);
    } catch (std::exception &e) {
      std::cerr
          << "Error happened during writing data into sycl-config-temp.txt: "
          << e.what() << std::endl;
      return 2;
    }

    if (file.bad() || file.fail()) {
      std::cerr << "Error happened during writing data into sycl-config-temp.txt!"
                << std::endl;
      return 3;
    }
  }

  // Make sure that the right config file will be read
  setenv("SYCL_CONFIG_FILE_NAME", "sycl-config-temp.txt", 1);

  try {
    sycl::detail::readConfig(/* ForceInitialization = */ true);
  } catch (sycl::exception &e) {
    // There are format requirements for the config file. To make sure that no
    // false positives are emitted if randomly-generated input looks _like_ a
    // valid config, but violate format requirements, we ignore certain
    // exceptions here.
    if (!contains(e.what(), "SPACE found at the beginning/end") &&
        !contains(e.what(), "The value contains more than") &&
        !contains(e.what(), "Variable name is more than"))
      throw e;
  }

  return 0;
}
