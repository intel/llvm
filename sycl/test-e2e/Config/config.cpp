//==---- config.cpp --------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %clangxx %debug_option -O0 -fsycl %s -o %t.out
// RUN: echo "SYCL_PRINT_EXECUTION_GRAPH=always" > %t.cfg
// RUN: env SYCL_CONFIG_FILE_NAME=%t.cfg %t.out
// RUN: ls | grep dot
// RUN: rm *.dot
// RUN: env SYCL_PRINT_EXECUTION_GRAPH=always %t.out
// RUN: ls | grep dot
// RUN: rm *.dot
// RUN: %t.out
// RUN: ls | not grep dot

#include <sycl/sycl.hpp>

int main() {
  sycl::buffer<int, 1> Buf(sycl::range<1>{1});
  auto Acc = Buf.get_access<sycl::access::mode::read>();
}
