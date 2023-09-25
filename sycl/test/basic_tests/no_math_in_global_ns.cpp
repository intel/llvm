// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note
// expected-no-diagnostics

// MSVC has the following includes:
// <ostream>
//   -> <ios>
//     -> <xlocnum>
//       -> <cmath>
// XFAIL: windows

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.single_task([=] { sqrt(1.0); }).wait();
  return 0;
}
