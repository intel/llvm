// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-2017-compat -fsyntax-only %s -internal-isystem %S/Inputs

// This test verifies that we generate deferred diagnostics when
// such diagnostics are in a function template.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

int global_value = -1;

template <typename T>
void kernel_wrapper() {
  q.submit([&](handler &h) {
    h.single_task([=] {
      // expected-error@+1{{SYCL kernel cannot use a non-const global variable}}
      (void)global_value;
    });
  });
}

int main() {
  kernel_wrapper<int>();
}
