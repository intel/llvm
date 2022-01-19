// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-2017-compat -fsyntax-only %s -internal-isystem %S/Inputs

// This test verifies that we generate deferred diagnostics when
// such diagnostics are in a function template.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;
template <typename h>
h *malloc_shared();

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
  int *x = malloc_shared<int>();
  kernel_handler kh;
  q.submit([&](handler &h) {
    h.single_task<class mykern>([=](auto g) {
      // expected-error@+1{{SYCL kernel cannot use a non-const global variable}}
      x[3] = global_value;
    },
                                kh);
  });
  kernel_wrapper<int>();
}
