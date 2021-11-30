// RUN: %clangxx -std=c++14 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify=expected,cxx14 %s -c -o %t.out
// RUN: %clangxx -std=c++14 -fsycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out -DSYCL_DISABLE_CPP_VERSION_CHECK_WARNING=1
// RUN: %clangxx -std=c++14 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify=cxx14,warning_extension,expected %s -c -o %t.out
// RUN: %clangxx -std=c++17 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++20 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx            -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out

// The test checks SYCL headers C++ compiance and that a warning is emitted
// when compiling in < C++17 mode.

#include <CL/sycl.hpp>

// cxx14-warning@* {{DPCPP does not support C++ version earlier than C++17. Some features might not be available.}}
//
// The next warning is not emitted in device compilation for some reason
// warning_extension-warning@* 0-1 {{#warning is a language extension}}
//
// The next warning is emitted for windows only
// expected-warning@* 0-1 {{Alignment of class vec is not in accordance with SYCL specification requirements, a limitation of the MSVC compiler(Error C2719).Requested alignment applied, limited at 64.}}

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
