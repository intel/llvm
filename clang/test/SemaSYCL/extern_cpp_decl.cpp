// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only %s

// Verify that declaration context validation techniques account for
// header files wrapped in extern "C++" declaration blocks.

// expected-no-diagnostics

extern "C++" {
#include "Inputs/sycl.hpp"
}

using namespace cl::sycl;

int main() {
  accessor<int, 1, access::mode::read_write> ok_acc;

  kernel_single_task<class use_local>(
      [=]() {
        ok_acc.use();
      });
}
