// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks that get_local_linear_id is the same on the nd_item as on the
// corresponding group.

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  const sycl::range<3> GlobalRange(2, 8, 16);
  const sycl::range<3> LocalRange(2, 4, 4);
  sycl::queue Q;
  sycl::buffer<bool> ReadSame_buf{GlobalRange.size()};
  Q.submit([&](sycl::handler &h) {
    sycl::accessor ReadSame{ReadSame_buf, h};
    h.parallel_for(sycl::nd_range<3>{GlobalRange, LocalRange},
                   [=](sycl::nd_item<3> Item) {
                     ReadSame[Item.get_global_linear_id()] =
                         Item.get_local_linear_id() ==
                         Item.get_group().get_local_linear_id();
                   });
  });
  sycl::host_accessor ReadSame{ReadSame_buf};
  int Failures = 0;
  for (size_t I = 0; I < GlobalRange.size(); ++I) {
    if (ReadSame[I])
      continue;
    ++Failures;
    std::cout << "Read mismatch at index " << I << std::endl;
  }
  return Failures;
}
