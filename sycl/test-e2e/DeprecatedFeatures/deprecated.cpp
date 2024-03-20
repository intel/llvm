// RUN: %{build} -D__SYCL_INTERNAL_API -o %t.out
// RUN: %{run} %t.out

//==------------- deprecated.cpp - SYCL 2020 deprecation test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  device Device{default_selector_v};
  platform Platform{default_selector_v};

  bool b = Device.has_extension("cl_intel_subgroups");
  b = Platform.has_extension("some_extension");

  return 0;
}
