// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -fno-gpu-rdc -internal-isystem %S/Inputs %s

// Check that uses of SYCL_EXTERNAL throw an error if -fno-gpu-rdc is passed
#include "sycl.hpp"

// expected-error@+1{{unknown type name 'SYCL_EXTERNAL'}}
SYCL_EXTERNAL void syclExternal() {}

using namespace sycl;
queue q;

void kernel_wrapper() {
  q.submit([&](handler &h) {
    h.single_task([=] {
    });
  });
}

int main() {
  kernel_wrapper();
}
