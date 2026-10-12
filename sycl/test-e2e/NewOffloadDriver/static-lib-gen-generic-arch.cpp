//==- static-lib-gen-generic-arch.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Link an object built for bmg against a static library whose member
// was built for spir64_gen with no device specified, using the new offloading
// model. The bmg wrapper object must also define the generic code's
// __sycl_registerlib_<unique_identifier> in addition to its own.

// REQUIRES: ocloc, target-spir, linux, arch-intel_gpu_bmg_g21

// DEFINE: %{ond} = --offload-new-driver -Wno-error=unused-command-line-argument

// Notice no device is specified in the below compile command.
// RUN: %clangxx -fsycl %{ond} -fsycl-targets=spir64_gen -c %S/Inputs/static-lib-registerlib-gen-generic.cpp -o %t.gen.o
// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t.gen.o

// RUN: %clangxx -fsycl %{ond} -fsycl-targets=intel_gpu_bmg_g21 -c %s -o %t.main.o
// RUN: %clangxx -fsycl %{ond} -fsycl-targets=intel_gpu_bmg_g21 %t.main.o %t.a -o %t.exe
// RUN: %{run} %t.exe

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cstddef>
#include <iostream>

#include "Inputs/static-lib-registerlib.hpp"

int main() {
  constexpr std::size_t N = 8;
  sycl::queue q;
  std::size_t *p = sycl::malloc_shared<std::size_t>(N, q);
  q.parallel_for<class KernelBMG>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
     p[idx] = idx[0];
   }).wait();
  std::size_t out[N] = {};
  run_gen_generic(out, N);

  bool Ok = true;
  for (std::size_t I = 0; I < N; ++I) {
    if (p[I] != I || out[I] != I + GenGenericOffset) {
      std::cout << "fail: p[" << I << "] == " << p[I] << ", out[" << I
                << "] == " << out[I] << "\n";
      Ok = false;
    }
  }
  sycl::free(p, q);
  if (Ok)
    std::cout << "pass\n";
  return Ok ? 0 : 1;
}
