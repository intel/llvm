// RUN: %clangxx -std=c++17 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s
// RUN: %clangxx -std=c++20 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s
// RUN: %clangxx            -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s

// The test checks SYCL headers C++ compliance with C++17 and later standards.

#include <sycl/sycl.hpp>

// clang-format off
// The next warning is emitted for windows only
// expected-warning@* 0-1 {{Alignment of class vec is not in accordance with SYCL specification requirements, a limitation of the MSVC compiler(Error C2719).Requested alignment applied, limited at 64.}}
// clang-format on

class KernelName1;

int main() {

  // Simple code to trigger instantiation of basic classes
  sycl::buffer<int, 1> Buf(sycl::range<1>{42});

  sycl::queue Q;

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::read>(CGH);

    CGH.parallel_for<KernelName1>(sycl::range<1>{42},
                                  [=](sycl::id<1> ID) { (void)Acc; });
  });
  Q.wait();
}
