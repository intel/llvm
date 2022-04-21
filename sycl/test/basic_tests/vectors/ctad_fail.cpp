// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected %s
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main() {
  sycl::vec v(1, .1); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'vec'}}
}
