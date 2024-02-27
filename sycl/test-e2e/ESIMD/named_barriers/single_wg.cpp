//==-------------- single_wg.cpp - DPC++ ESIMD on-device test -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Test checks support of named barrier in ESIMD kernel.
// Basic case with 1 work-group and 16 threads: 4 producer and 12 consumer.
// SLM and surface size is 64 bytes.
// Producers store data to SLM, then all threads read SLM and store data to
// surface.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num> class KernelID;

template <int case_num, unsigned prods, unsigned cons, class QueueTY>
bool test(QueueTY q) {
  constexpr unsigned Groups = 1;
  constexpr unsigned Threads = cons + prods;
  constexpr unsigned Size = 4 * Threads;
  constexpr unsigned scale = cons / prods;

  static_assert(cons >= prods, "Consumers must be greater than producers");
  static_assert(cons % prods == 0, "Consumers must be multiple of producers");
  static_assert(Threads > 3, "Total number of threads must be greater than 3");

  std::vector<int> out(Size, 0);

  try {
    buffer<int, 1> buf(out.data(), out.size());

    // workgroups
    sycl::range<1> GlobalRange{Groups};
    // threads in each group
    sycl::range<1> LocalRange{Threads};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            // 1 named barrier, id 0 reserved for unnamed
            constexpr unsigned bnum = 2;
            constexpr unsigned bid = 1;

            // number of ints loaded/stored by each thread
            constexpr unsigned VL = Size / Threads;
            // number of ints stored to SLM by producer
            constexpr unsigned NUM = VL * (1 + scale);

            named_barrier_init<bnum>();

            unsigned int idx = ndi.get_local_id(0);
            unsigned int off = idx * VL * sizeof(int);

            bool is_producer = (idx % (scale + 1)) == scale;
            bool is_consumer = !is_producer;
            // only-consumer or only-producer modes
            unsigned int flag = is_producer ? 0x1 : 0x2;

            slm_init(Size * sizeof(int));
            slm_block_store(off, simd<int, VL>(0));
            barrier();

            if (is_producer) {
              unsigned int x = VL * (idx - scale);
              unsigned int p_off = x * sizeof(int);
              // each producer stores x4 data
              simd<int, NUM> init(0xdead0000 + x, 1);
              slm_block_store(p_off, init); // producers store data to SLM
            }

            // signaling after data stored
            named_barrier_signal(bid, flag, prods, cons);

            if (is_consumer)
              named_barrier_wait(bid); // consumers waiting for signal

            auto val = slm_block_load<int, VL>(off); // reading SLM
            lsc_block_store<int, VL>(acc, off,
                                     val); // and storing it to output surface
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return -1;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0000 + i;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }

  std::cout << "#" << case_num << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}

int main() {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(q);

  bool passed = true;

  passed &= test<1, 4, 12>(q);
  passed &= test<2, 2, 14>(q);
  passed &= test<3, 8, 24>(q);
  passed &= test<4, 2, 2>(q);

  return passed ? 0 : 1;
}
