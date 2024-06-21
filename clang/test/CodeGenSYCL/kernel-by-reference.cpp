// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -sycl-std=2020 %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -sycl-std=2020 -Wno-sycl-strict %s

#include "sycl.hpp"

using namespace sycl;

// expected-no-diagnostics

int simple_add(int i) {
  return i + 1;
}

// ensure both compile.
int main() {
  queue q;

  q.submit([&](handler &h) {
    h.single_task<class sycl2020>([]() { simple_add(11); });
  });

  return 0;
}
