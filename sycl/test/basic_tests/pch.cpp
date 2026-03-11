// RUN: %clang -fsycl -c -x c++-header %S/head.hpp -o %S/head.pch
// RUN: %clang -fsycl -c -include-pch %S/head.pch %s

// Verify a PCH file created from the header file can be
// successfully included in a source file compilation.

// expected-no-diagnostics

#include "head.hpp"

using namespace std;
int main() {
  sycl::range<3> three_dim_range(64, 1, 2);
  assert(three_dim_range.size() == 128);
  cout << "three_dim_range passed " << endl;
}
