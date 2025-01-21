// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

// Tests that read-only accessors can be used in multi_ptr deduction guides.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  int Data = 0;
  sycl::buffer<int, 1> Buf{&Data, {1}};

  Q.submit([&](sycl::handler &CGH) {
     sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::device> Acc(
         Buf, CGH);
     CGH.single_task([=] { auto MPtr = sycl::multi_ptr(Acc); });
   }).wait_and_throw();
  return 0;
}
