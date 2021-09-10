// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2017 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=1.2.1 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

int main() {
    cl::sycl::id<1> id_obj(64);
    // expected-warning@+1 {{'operator range' is deprecated: range() conversion is deprecated}}
    (void)(cl::sycl::range<1>)id_obj;

    return 0;
}