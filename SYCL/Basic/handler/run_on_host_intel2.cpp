// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

// This tests that early free of command (and, hence, the command group) won't
// affect "native kernel" feature support.
int main(void) {
  sycl::queue Q;

  int *Ptr = new int;

  auto E = Q.submit(
      [&](sycl::handler &CGH) { CGH.run_on_host_intel([=] { *Ptr = 5; }); });

  E.wait();

  std::cout << "Finished successfully\n";

  return 0;
}
