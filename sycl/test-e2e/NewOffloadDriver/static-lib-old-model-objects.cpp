// Link static libraries holding old offloading model objects, with and without
// objects pre-linked with -fsycl-link, using the new offloading model.

// REQUIRES: target-spir, linux

// RUN: %clangxx -fsycl --no-offload-new-driver -c %S/../AOT/Inputs/static-lib-registerlib-jit.cpp -o %t.jit.o
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-link %t.jit.o -o %t.jit.pre.o
// RUN: rm -f %t.a %t.pre.a
// RUN: llvm-ar crv %t.a %t.jit.o
// RUN: llvm-ar crv %t.pre.a %t.jit.o %t.jit.pre.o

// The new offloading model passes different device compile options by default.
// DEFINE: %{ond} = -fsycl --offload-new-driver -ftarget-register-alloc-mode=pvc:default
// RUN: %clangxx %{ond} %s %t.a -o %t.exe
// RUN: %{run} %t.exe
// RUN: %clangxx %{ond} %s %t.pre.a -o %t.pre.exe
// RUN: %{run} %t.pre.exe

#include <sycl/detail/core.hpp>

#include <cstddef>
#include <iostream>

#include "../AOT/Inputs/static-lib-registerlib.hpp"

int main() {
  constexpr std::size_t N = 8;
  std::size_t Out[N] = {};
  run_jit(Out, N);
  for (std::size_t I = 0; I < N; ++I)
    if (Out[I] != I + JITOffset) {
      std::cout << "fail: Out[" << I << "] == " << Out[I] << "\n";
      return 1;
    }
  std::cout << "pass\n";
  return 0;
}
