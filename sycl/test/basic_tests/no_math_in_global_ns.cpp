// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.single_task([=] { sqrt(1.0); }).wait();
  return 0;
}
