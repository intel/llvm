#include <sycl/detail/core.hpp>

#include "sycl/ext/intel/fpga_extensions.hpp"
#include <sycl/usm.hpp>

#include <iostream>

#include "fpga_kernels.h"

int main() {
  sycl::queue q;
  int *result = sycl::malloc_host<int>(13, q);
  if (!result)
    std::cout << "Error: failed to allocate USM host memory\n";

  /////////////////////////////////////////////////////////////////////////
  // Kernel add
  /////////////////////////////////////////////////////////////////////////
  try {
    add(q, &(result[0]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel add_x_1
  /////////////////////////////////////////////////////////////////////////
  try {
    add_x_1(q, &(result[1]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add_x_1 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel add_x_2
  /////////////////////////////////////////////////////////////////////////
  try {
    add_x_2(q, &(result[2]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add_x_2 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel add_x_3
  /////////////////////////////////////////////////////////////////////////
  try {
    add_x_3(q, &(result[3]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add_x_3 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel add_x_4
  /////////////////////////////////////////////////////////////////////////
  try {
    add_x_4(q, &(result[4]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add_x_4 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel add_x_5
  /////////////////////////////////////////////////////////////////////////
  try {
    add_x_5(q, &(result[5]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching add_x_5 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }

  /////////////////////////////////////////////////////////////////////////
  // Kernel sub
  /////////////////////////////////////////////////////////////////////////
  try {
    sub(q, &(result[6]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_1
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_1(q, &(result[7]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_1 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_2
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_2(q, &(result[8]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_2 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_3
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_3(q, &(result[9]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_3 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_4
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_4(q, &(result[10]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_4 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_5
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_5(q, &(result[11]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_5 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////////////////
  // Kernel sub_x_6
  /////////////////////////////////////////////////////////////////////////
  try {
    sub_x_6(q, &(result[12]), 2, 1);
  } catch (sycl::exception const &e) {
    std::cout
        << "Caught synchronous SYCL exception while launching sub_x_6 kernel:\n"
        << e.what() << "\n";
    std::terminate();
  }

  q.wait();

  // Check the results
  if ( // add kernels
      result[0] == 3 && result[1] == 4 && result[2] == 5 && result[3] == 6 &&
      result[4] == 7 && result[5] == 8 &&
      // sub kernels
      result[6] == 1 && result[7] == 0 && result[8] == -1 && result[9] == -2 &&
      result[10] == -3 && result[11] == -4 && result[12] == -5)
    std::cout << "PASSED\n";
  else {
    std::cout << "FAILED\n";
    exit(-1);
  }
}
