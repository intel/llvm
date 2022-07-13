// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

//==-- run_on_host_intel.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include "../../helpers.hpp"

using namespace cl;

template <typename SrcAccType, typename DstAccType>
void copyAndAdd(SrcAccType SrcAcc, DstAccType DstAcc, int Var) {
  for (int I = 0; I < (int)DstAcc.get_count(); ++I)
    DstAcc[I] = Var + SrcAcc[I];
}

int main() {
  constexpr size_t BufSize = 4;
  int data1[BufSize] = {-1, -1, -1, -1};
  sycl::buffer<int, 1> SrcBuf(data1, sycl::range<1>{BufSize});
  sycl::buffer<int, 1> DstBuf(sycl::range<1>{BufSize});

  TestQueue Queue{sycl::default_selector{}};
  Queue.submit([&](sycl::handler &CGH) {
    auto SrcAcc = SrcBuf.get_access<sycl::access::mode::read>(CGH);
    auto DstAcc = DstBuf.get_access<sycl::access::mode::write>(CGH);
    const int Var = 43;

    CGH.run_on_host_intel([=]() { copyAndAdd(SrcAcc, DstAcc, Var); });
  });

  auto DstAcc = DstBuf.template get_access<sycl::access::mode::read_write>();
  const int Expected = 42;
  for (int I = 0; I < DstAcc.get_count(); ++I)
    if (DstAcc[I] != Expected) {
      std::cerr << "Mismatch. Elem " << I << ". Expected: " << Expected
                << ", Got: " << DstAcc[I] << std::endl;
      return 1;
    }

  std::cout << "Success" << std::endl;

  return 0;
}
