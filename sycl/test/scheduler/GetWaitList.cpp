// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: %t.out
//==------------------- GetWaitList.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  sycl::event start;
  start.wait_and_throw();
}
