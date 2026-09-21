//==- static-lib-multiple-objects-multiple-targets.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Pull SYCL device code out of a static library, without --whole-archive, using
// the new offloading model and gcc for the final host link. The archive holds
// three translation units, each pre-linked with -fsycl-link for a single
// distinct target (spir64 / intel_gpu_dg2 / intel_gpu_bmg_g21), so each
// contributes its own __sycl_registerlib_<hash>.
//
// The Windows counterpart is
// static-lib-multiple-objects-multiple-targets-win.cpp.

// REQUIRES: ocloc, target-spir, linux
// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12 || arch-intel_gpu_bmg_g21

// DEFINE: %{ond} = --offload-new-driver -Wno-error=unused-command-line-argument

// RUN: %clangxx -fsycl %{ond} -fsycl-targets=spir64 -fPIC -c %S/Inputs/static-lib-registerlib-jit.cpp -o %t.jit.o
// RUN: %clangxx -fsycl %{ond} -fsycl-link -fsycl-targets=spir64 -fPIC %t.jit.o -o %t.jit.pre.o
// RUN: %clangxx -fsycl %{ond} -fsycl-targets=intel_gpu_dg2 -fPIC -c %S/Inputs/static-lib-registerlib-dg2.cpp -o %t.dg2.o
// RUN: %clangxx -fsycl %{ond} -fsycl-link -fsycl-targets=intel_gpu_dg2 -fPIC %t.dg2.o -o %t.dg2.pre.o
// RUN: %clangxx -fsycl %{ond} -fsycl-targets=intel_gpu_bmg_g21 -fPIC -c %S/Inputs/static-lib-registerlib-bmg.cpp -o %t.bmg.o
// RUN: %clangxx -fsycl %{ond} -fsycl-link -fsycl-targets=intel_gpu_bmg_g21 -fPIC %t.bmg.o -o %t.bmg.pre.o

// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t.jit.o %t.jit.pre.o %t.dg2.o %t.dg2.pre.o %t.bmg.o %t.bmg.pre.o

// RUN: %clangxx -Wno-error=unused-command-line-argument %sycl_include %cxx_std_optionc++17 -fPIC -c %s -o %t.main.o
// RUN: g++ %t.main.o %t.a -L%sycl_libs_dir -lsycl -Wl,-rpath,%sycl_libs_dir -o %t.exe
// RUN: %{run} %t.exe

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

#include <cstddef>
#include <iostream>

#include "Inputs/static-lib-registerlib.hpp"

namespace syclex = sycl::ext::oneapi::experimental;

static bool check(const char *Name, std::size_t Offset,
                  void (*Run)(std::size_t *, std::size_t)) {
  constexpr std::size_t N = 8;
  std::size_t out[N] = {};
  Run(out, N);
  for (std::size_t I = 0; I < N; ++I) {
    if (out[I] != I + Offset) {
      std::cout << "fail " << Name << ": out[" << I << "] == " << out[I]
                << ", expected " << (I + Offset) << "\n";
      return false;
    }
  }
  std::cout << Name << " ran\n";
  return true;
}

int main() {
  // The JIT (spir64) kernel runs on any SPIR device; the AOT kernels only have
  // an image for their target, so only invoke the one matching this device.
  bool Ok = check("jit", JITOffset, run_jit);
  auto Arch =
      sycl::queue{}.get_device().get_info<syclex::info::device::architecture>();
  using arch = syclex::architecture;
  switch (Arch) {
  case arch::intel_gpu_dg2_g10:
  case arch::intel_gpu_dg2_g11:
  case arch::intel_gpu_dg2_g12:
    Ok &= check("dg2", DG2Offset, run_dg2);
    break;
  case arch::intel_gpu_bmg_g21:
    Ok &= check("bmg", BMGOffset, run_bmg);
    break;
  default:
    std::cout << "no matching AOT kernel for this device\n";
    break;
  }

  if (Ok)
    std::cout << "pass\n";
  return Ok ? 0 : 1;
}
