//==---------------- preemption.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: linux
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_lnl_m
// RUN: %{build} -o %t.out
// RUN: env IGC_DumpToCustomDir=%t.dump NEO_CACHE_PERSISTENT=0 IGC_ShaderDumpEnable=1 %{run} %t.out
// RUN: %{run-aux} grep enablePreemption %t.dump/*.asm

// The test expects to see "enablePreemption" switch in the compilation
// switches. It fails if does not find it.

#include "esimd_test_utils.hpp"

using namespace sycl;

int main() {
  queue q;
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(sycl::range<1>{1},
                                 [=](id<1> id) SYCL_ESIMD_KERNEL {});
  });
  return 0;
}
