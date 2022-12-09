// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -fno-gpu-rdc -internal-isystem %S/Inputs %s

// Check that declarations of SYCL_EXTERNAL functions throw an error if -fno-gpu-rdc is passed
#include "sycl.hpp"

// expected-error@+1{{invalid declaration of SYCL_EXTERNAL function in non-relocatable device code mode}}
SYCL_EXTERNAL void syclExternalDecl();

// expected-error@+1{{invalid definition of SYCL_EXTERNAL function in non-relocatable device code mode}}
SYCL_EXTERNAL void syclExternalDefn() {}

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
