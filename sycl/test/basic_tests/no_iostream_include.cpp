// RUN: %clangxx -DSYCL2020_CONFORMANT_APIS=1 -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// Test that only iosfwd and not istream, ostream, and iostream_proxy
// are included.
#include <sycl/sycl.hpp>

void t(std::istream &i, std::ostream &o) {
  char c;
  i >> c;   // expected-error {{invalid operands to binary expression}}
  o << o;   // expected-error {{invalid operands to binary expression}}
  std::cout // expected-error {{no member named 'cout' in namespace 'std'}}
      << "\n";
}
