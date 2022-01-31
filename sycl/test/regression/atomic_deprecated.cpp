// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>
int main() {
  cl::sycl::multi_ptr<int, cl::sycl::access::address_space::global_space> a(
      nullptr);
  // expected-warning@+1 {{'atomic<int, sycl::access::address_space::global_space>' is deprecated: sycl::atomic is deprecated since SYCL 2020}}
  cl::sycl::atomic<int> b(a);
  return 0;
}
