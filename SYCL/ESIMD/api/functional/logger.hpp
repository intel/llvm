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

namespace esimd_test::api::functional {

// Interface for any test case description class to use for logs within generic
// assertions
struct ITestDescription {
  virtual ~ITestDescription() = default;
  virtual std::string to_string() const = 0;
};

namespace log {

// Printing failure message to console with pre-defined format
inline void fail(const ITestDescription &test_description) {
  // Force output buffer flush after each failure
  std::cout << test_description.to_string() << std::endl;
}

// Printing provided string to the console with a forced flush to have all logs
// available in case of `abort()` because of a test crash
inline void note(const std::string &msg) { std::cout << msg << std::endl; }

} // namespace log

} // namespace esimd_test::api::functional
