// REQUIRES: gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------- local_mem_type.cpp - SYCL local mem type test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {
  assert(device().get_info<info::device::local_mem_type>() !=
         info::local_mem_type::none);
}
