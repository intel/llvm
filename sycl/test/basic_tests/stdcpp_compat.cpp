// RUN: %clangxx -std=c++14 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify-ignore-unexpected=error,note,warning -Xclang -verify=expected,cxx14 %s -c -o %t.out
// RUN: %clangxx -std=c++14 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify-ignore-unexpected=error,note,warning -Xclang -verify=cxx14,warning_extension,expected  %s -c -o %t.out
// RUN: %clangxx -std=c++17 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++20 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx            -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out

// The test checks SYCL headers C++ compiance and that a warning is emitted
// when compiling in < C++17 mode.

#include <sycl/sycl.hpp>

// clang-format off
// cxx14-error@* {{static assertion failed due to requirement '201402L >= 201703L'}}
//
// The next warning is not emitted in device compilation for some reason
// warning_extension-warning@* 0-1 {{#warning is a C++2b extension}}
//
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
