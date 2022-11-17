// RUN: %clangxx %s %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning
#include <sycl/range.hpp>

using namespace std;
int main() {
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::range<1>'}}
  sycl::range<1> one_dim_range_f1(64, 2, 4);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::range<2>'}}
  sycl::range<2> two_dim_range_f1(64);
}
