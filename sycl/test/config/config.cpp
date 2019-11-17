//==---- config.cpp --------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %clangxx -g -O0 -fsycl %s -o %t1.out
// RUN: %clangxx -g -O0 -fsycl %s -o %t2.out
// RUN: echo "SYCL_PRINT_EXECUTION_GRAPH=ValueFromConfigFile" > %t3.cfg
// RUN: env SYCL_CONFIG_FILE=%t3.cfg env SYCL_PRINT_EXECUTION_GRAPH=VlaueFromEnvVar %t1.out
// RUN: env SYCL_CONFIG_FILE=%t3.cfg %t1.out
// RUN: %t2.out

#include <CL/sycl/detail/config.hpp>

#include <cstring>

int main() {
  const char *Val = cl::sycl::detail::SYCLConfig<
      cl::sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get();

  if (getenv("SYCL_PRINT_EXECUTION_GRAPH")) {
    if (!Val)
      return 1;
    return strcmp(Val, "VlaueFromEnvVar");
  }

  if (getenv("SYCL_CONFIG_FILE")) {
    if(!Val)
      return 1;
    return strcmp(Val, "ValueFromConfigFile");
  }

  return nullptr != Val;
}
