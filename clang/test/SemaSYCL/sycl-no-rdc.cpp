// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -fsycl-allow-func-ptr -fno-gpu-rdc -internal-isystem %S/Inputs %s

// Check that accesses to undefined SYCL_EXTERNAL functions throw an error if -fno-gpu-rdc is passed
#include "sycl.hpp"

SYCL_EXTERNAL void syclExternalUndefined();

SYCL_EXTERNAL void syclExternalDefined() {}

using namespace sycl;
queue q;

void kernel_wrapper() {
  q.submit([&](handler &h) {
    h.single_task([=] {
     // expected-error@+1{{separate compilation unit without relocatable device code}}
     syclExternalUndefined();
     syclExternalDefined();
     // expected-error@+1{{separate compilation unit without relocatable device code}}
     auto fcnPtr = 1 == 0 ? syclExternalUndefined : syclExternalDefined;
     fcnPtr();
     // expected-error@+1{{separate compilation unit without relocatable device code}}
     constexpr auto constExprFcnPtr = 1 == 0 ? syclExternalUndefined : syclExternalDefined;
     constExprFcnPtr();
    });
  });
}

int main() {
  kernel_wrapper();
}
