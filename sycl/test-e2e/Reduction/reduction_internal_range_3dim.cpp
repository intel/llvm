// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);

  testRange(Storage, range<3>{7, 7, 5});

  return 0;
}
