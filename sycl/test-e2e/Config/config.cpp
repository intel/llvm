//==---- config.cpp --------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} %debug_option -O0 -o %t.out
// RUN: echo SYCL_PRINT_EXECUTION_GRAPH=always > %t.cfg
// RUN: env SYCL_CONFIG_FILE_NAME=%t.cfg %t.out
// RUN: cat *.dot > /dev/null
// RUN: rm *.dot
// RUN: env SYCL_PRINT_EXECUTION_GRAPH=always %t.out
// RUN: cat *.dot > /dev/null
// RUN: rm *.dot
// RUN: %t.out
// RUN: not cat *.dot > /dev/null

#include <sycl/detail/core.hpp>

int main() {
  sycl::buffer<int, 1> Buf(sycl::range<1>{1});
  sycl::host_accessor Acc(Buf, sycl::read_only);
  return 0;
}
