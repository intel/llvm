// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

int main() {

  long long int a = -9223372034707292160ll;
  long long int b = sycl::ctz(a);
  // expected-warning@+1 {{'ctz<long long>' is deprecated: 'sycl::ext::intel::ctz' is deprecated, use 'sycl::ctz' instead}}
  long long int c = sycl::ext::intel::ctz(a);

  return 0;
}
