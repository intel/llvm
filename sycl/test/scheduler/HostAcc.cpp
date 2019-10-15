// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_THROW_ON_BLOCK=1 %t.out
//==--------------------------- HostAcc.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl;

int main() {

  bool Fail = false;
  // Check that multiple host accessor which are created with read access mode
  // can exist simultaneously.
  sycl::buffer<int, 1> Buf(sycl::range<1>{1});
  {
    auto Acc1 = Buf.get_access<sycl::access::mode::read>();
    auto Acc2 = Buf.get_access<sycl::access::mode::read>();
  }

  // Check creation of write accessor after read blocks and generate exception
  // (special env var is set for throwing exception) caught
  {
    bool ExcCaught = false;
    try {
      auto Acc1 = Buf.get_access<sycl::access::mode::read>();
      auto Acc3 = Buf.get_access<sycl::access::mode::write>();
    } catch (sycl::runtime_error &E) {
      ExcCaught = true;
    }
    Fail |= !ExcCaught;
  }

  return Fail;
}
