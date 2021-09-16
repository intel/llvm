// RUN: %clangxx -std=c++14 -fsycl --no-system-header-prefix=CL/sycl -Wall -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify=bad %s -c -o %t.out
// RUN: %clangxx -std=c++17 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++20 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx            -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out

// expected-no-diagnostics

#include <CL/sycl.hpp> // bad-warning@* {{DPCPP does not support C++ version earlier than C++17. Some features might not be available.}}

class KernelName1;

int main() {

  // Simple code to trigger instantiation of basic classes
  sycl::buffer<int, 1> Buf(sycl::range<1>{42});

  sycl::queue Q;

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<cl::sycl::access::mode::read>(CGH);

    CGH.parallel_for<KernelName1>(sycl::range<1>{42},
                                  [=](sycl::id<1> ID) { (void)Acc; });
  });
  Q.wait();
}
