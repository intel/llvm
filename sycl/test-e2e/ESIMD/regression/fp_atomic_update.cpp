//==----------- fp_atomic_update.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <bit>
#include <bitset>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

int main() {
  sycl::queue queue{gpu_selector_v};
  esimd_test::printTestLabel(queue);
  constexpr size_t N = 8;
  double *data_d = sycl::malloc_device<double>(N, queue);
  double *out_d = sycl::malloc_shared<double>(N, queue);
  int errCount = 0;

  try {

    std::vector<double> data(
        N, sycl::bit_cast<double>(uint64_t(0x400000018FFFFFFF)));

    queue.memcpy(data_d, data.data(), N * sizeof(double)).wait();
    queue.fill(out_d, sycl::bit_cast<double>(uint64_t(0x0000000000000001)), N)
        .wait();

    queue.parallel_for(sycl::nd_range<1>(1, 1),
                       [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                         // Atomically update the maximum value
                         simd<double, 8> tmp;
                         tmp.copy_from(data_d);
                         atomic_update<atomic_op::fmax>(
                             out_d, simd<uint32_t, N>(0, sizeof(double)), tmp);
                       });

    queue.wait_and_throw();

    std::vector<double> out_data(N);
    queue.memcpy(out_data.data(), out_d, N * sizeof(double)).wait();
    for (int iter = 0; iter < out_data.size(); iter++) {
      double relError = (out_data[iter] - data[iter]) / data[iter];
      if (relError != 0 && ++errCount < 10)
        std::cout << "ERROR at index " + std::to_string(iter) << ": "
                  << std::to_string(relError) + " != 0\n";
    }
  } catch (sycl::exception &e) {
    free(data_d, queue);
    free(out_d, queue);
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }
  free(data_d, queue);
  free(out_d, queue);
  std::cout << (errCount == 0 ? "Passed\n" : "Failed\n");
  return errCount != 0;
}
