// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Checks that get_local_linear_id is the same on the nd_item as on the
// corresponding group.

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  const sycl::range<3> GlobalRange(2, 8, 16);
  const sycl::range<3> LocalRange(2, 4, 4);
  sycl::queue Q;
  bool *ReadSame = sycl::malloc_shared<bool>(GlobalRange.size(), Q);
  Q.parallel_for(sycl::nd_range<3>{GlobalRange, LocalRange},
                 [=](sycl::nd_item<3> Item) {
                   ReadSame[Item.get_global_linear_id()] =
                       Item.get_local_linear_id() ==
                       Item.get_group().get_local_linear_id();
                 })
      .wait();
  int Failures = 0;
  for (size_t I = 0; I < GlobalRange.size(); ++I) {
    if (ReadSame[I])
      continue;
    ++Failures;
    std::cout << "Read mismatch at index " << I << std::endl;
  }
  sycl::free(ReadSame, Q);
  return Failures;
}
