// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);

  testRange(Storage, range<1>{42});
  // Exercise work-group reuse and a non-uniform tail for every strategy.
  size_t WGSize = 0, Groups = 0;
  q.submit([&](handler &cgh) {
    WGSize = detail::reduGetPreferredWGSize(cgh, sizeof(T));
    Groups = detail::reduGetMaxNumConcurrentWorkGroups(cgh);
  });
  testRange(Storage, range<1>{Groups * WGSize * 9 + 17});

  return 0;
}
