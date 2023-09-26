// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note
// expected-no-diagnostics

// MSVC has the following includes:
// <ostream>
//   -> <ios>
//     -> <xlocnum>
//       -> <cmath>
//
// <functional>
//   -> <unordered_map>
//     -> <xhash>
//       -> <cmath>
//
// <vector> and <string> seem to include <cmath> implicitly as well.
// XFAIL: windows

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.single_task([=] { sqrt(1.0); }).wait();
  return 0;
}
