// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected %s
#include <sycl/sycl.hpp>

int main() {
  sycl::vec v(1, .1); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'sycl::vec'}}
}
