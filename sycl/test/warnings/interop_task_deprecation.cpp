// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;
  buffer<int, 1> B(range<1>{1});
  Q.submit([&](handler &H) {
    auto Acc = B.get_access<access::mode::read>(H);
    // expected-warning@+1 {{interop_handler class is deprecated, use interop_handle instead with host-task}}
    auto L = [Acc](interop_handler IH) {
      // empty
    };
    // expected-warning@+2 {{interop_task() is deprecated, use host_task() instead}}
    // expected-warning@+1 {{interop_task() is deprecated, use host_task() instead}}
    H.interop_task(L);
  });
  return 0;
}
