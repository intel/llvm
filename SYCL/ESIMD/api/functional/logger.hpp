//===-- logger.hpp - Define functions for print messages to console. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the functions that provides easier
/// way to printing messages to the console.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>
#include <string>

namespace esimd_test {
namespace api {
namespace functional {
namespace log {

// Printing failure message to console with pre-defined format
template <int NumElems = 0>
inline void fail(const std::string &msg,
                 std::string underlying_data_type = "") {
  std::string log_msg{msg};
  if (!underlying_data_type.empty()) {
    log_msg += ", with data type: ";
    log_msg += underlying_data_type;
  }
  if constexpr (NumElems != 0) {
    log_msg += ", with num elements: ";
    log_msg += std::to_string(NumElems);
  }
  std::cout << log_msg << std::endl;
}

// Printing provided string to the console with a forced flush to have all logs
// available in case of `abort()` because of a test crash
inline void note(const std::string &msg) { std::cout << msg << std::endl; }

} // namespace log
} // namespace functional
} // namespace api
} // namespace esimd_test
