// Test early-AOT behaviors with -fsycl -fno-sycl-rdc.  This targets spir64_gen

// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda, hip

// Build the early AOT device binaries
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -fno-sycl-rdc -c -DADD_CPP %s -o %t_add.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -fno-sycl-rdc -c -DSUB_CPP %s -o %t_sub.o
// RUN: %clangxx -fsycl -DMAIN_CPP %s %t_add.o %t_sub.o -o %t.out

// RUN: %{run} %t.out

#ifdef MAIN_CPP
// main.cpp

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

#include <iostream>

void add(sycl::queue q, int *result, int a, int b);
void sub(sycl::queue q, int *result, int a, int b);

int main() {
  sycl::queue q;
  int *result = sycl::malloc_host<int>(2, q);
  if (!result)
    std::cout << "Error: failed to allocate USM host memory\n";

  try {
    add(q, &(result[0]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  try {
    sub(q, &(result[1]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  q.wait();

  // Check the results
  if (!(result[0] == 3 && result[1] == 1)) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}

#endif // MAIN_CPP

#ifdef ADD_CPP
// add.cpp
#include <sycl/detail/core.hpp>

void add(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_dummy>([=] { *result = a + b; });
}

#endif // ADD_CPP

#ifdef SUB_CPP
// sub.cpp
#include <sycl/detail/core.hpp>

void sub(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_dummy>([=] { *result = a - b; });
}
#endif // SUB_CPP
