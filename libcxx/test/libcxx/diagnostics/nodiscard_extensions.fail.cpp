// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// GCC versions prior to 7.0 don't provide the required [[nodiscard]] attribute.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6

// Test that entities declared [[nodiscard]] as at extension by libc++, are
// only actually declared such when _LIBCPP_ENABLE_NODISCARD is specified.

// All entities to which libc++ applies [[nodiscard]] as an extension should
// be tested here and in nodiscard_extensions.pass.cpp. They should also
// be listed in `UsingLibcxx.rst` in the documentation for the extension.

// MODULES_DEFINES: _LIBCPP_ENABLE_NODISCARD
#define _LIBCPP_ENABLE_NODISCARD

#include <memory>

#include "test_macros.h"

int main(int, char**) {
  {
    // expected-error-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
    std::get_temporary_buffer<int>(1);
  }

  return 0;
}
