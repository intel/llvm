// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;
  buffer<int, 1> B(range<1>{1});
  Q.submit([](handler &H) {
    auto Acc = B.get_access<access::mode::read>(H);
    auto L = [Acc]() {
      // empty
    };
    // expected-warning@+1 {{run_on_host_intel() is deprecated, use host_task() instead}}
    H.run_on_host_intel(L);
  });
  return 0;
}
