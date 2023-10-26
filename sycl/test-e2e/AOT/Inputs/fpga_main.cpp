#include "sycl/sycl.hpp"
#include "sycl/ext/intel/fpga_extensions.hpp"
#include <iostream>

#include "kernels.h"

int main() {
  sycl::queue q;
  int *result = sycl::malloc_host<int>(2, q);
  if (!result)
    std::cout << "Error: failed to allocate USM host memory\n";
    
  try {
    add(q, &(result[0]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout << "Caught synchronous SYCL exception while launching add kernel:\n" << e.what() << "\n";
    std::terminate();
  }
  try {
    sub(q, &(result[1]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout << "Caught synchronous SYCL exception while launching sub kernel:\n" << e.what() << "\n";
    std::terminate();
  }
  q.wait();

  // Check the results
  if (result[0] == 3 && result[1] == 1)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
}
