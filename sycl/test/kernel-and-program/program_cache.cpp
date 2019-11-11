// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//==------------- program_cache.cpp - SYCL kernel/program test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

class Functor {
  public:
  void operator()() {
  }
};

int main() {
  cl::sycl::queue q;
  cl::sycl::program prog(q.get_context());
  prog.build_with_kernel_type<Functor>();

  auto *ctx = cl::sycl::detail::getRawSyclObjImpl(prog.get_context());
  return ctx->getCachedPrograms().size() != 1;
}
