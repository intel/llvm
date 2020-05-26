//==--- device-code-second-file.cpp --- Device code split unit tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device-code-split-lib.hpp"

void runKernel1FromFile2(cl::sycl::queue Q, cl::sycl::buffer<int, 1> Buf) {
    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File2Kern1>([=]() { Acc[0] = 3; });
    });
}
