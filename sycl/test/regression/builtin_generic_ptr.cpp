// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -Xclang -verify %s %}
// expected-no-diagnostics

// Regression test ensuring math builtins accept multi_ptr in the generic
// address space.

#include <sycl.hpp>

int main() {
  sycl::queue Q;
  Q.single_task([=]() {
    double I = -3.1415926 / 4;
    double D = 0.0;
    volatile double Res = sycl::sincos(
        I, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                    sycl::access::decorated::yes, double>(&D));
  });
  return 0;
}
