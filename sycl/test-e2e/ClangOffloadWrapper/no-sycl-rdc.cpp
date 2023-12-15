// Test clang-offload-wrapper with -fno-sycl-rdc

// REQUIRES: gpu && level_zero
// UNSUPPORTED: cuda, hip

// RUN: %clangxx -fsycl --offload-new-driver %s %S/Inputs/add.cpp %S/Inputs/sub.cpp -o %t1.out \
// RUN:  -Xdevice-post-link "-split=auto -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native" -v 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-RDC %s

// RUN: %{run} %t1.out

// RUN: %clangxx -fsycl -fno-sycl-rdc --offload-new-driver %s %S/Inputs/add.cpp %S/Inputs/sub.cpp -o %t2.out \
// RUN:  -Xdevice-post-link "-split=auto -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native" -v 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-NO-RDC %s

// RUN: %{run} %t2.out

// Should only be one call of sycl-post-link for RDC mode.
// CHECK-RDC: sycl-post-link{{.*}} -split=auto
// CHECK-RDC-NOT: sycl-post-link

// Should call sycl-post-link 3 times, once per each input file
// CHECK-NO-RDC-COUNT-3: sycl-post-link{{.*}} -split=auto
// CHECK-NO-RDC-NOT: sycl-post-link

#include "sycl/sycl.hpp"
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
