//=------------- loop_extended.cpp - DPC++ ESIMD on-device test -------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -o %t1.out -DEXP
// RUN: %{run} %t1.out
//
// Test checks support of named barrier in a loop in ESIMD kernel.
// First iteration has 1 barrier and 1 producer, second - 2 barriers and 2
// producers. Producer stores data to SLM, then all threads read SLM and store
// data to surface.

#include "../esimd_test_utils.hpp"

#ifdef EXP
#define NS __ESIMD_ENS
#else
#define NS __ESIMD_NS
#endif

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num> class KernelID;

template <unsigned case_num, unsigned Size, class QueueTY>
bool test(QueueTY q) {
  constexpr unsigned Groups = 1;
  constexpr unsigned Threads = 8;

  static_assert((Size % (2 * Threads) == 0) && (Size >= (2 * Threads)),
                "Inv case");

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
            // 2 named barriers, id 0 reserved for unnamed
            constexpr unsigned bnum = 3;

            // SLM size is half of output surface size so
            // content of SLM can be copied to out buffer on each iteration
            constexpr unsigned SlmSize = Size / 2;     // 32
            constexpr unsigned VL = SlmSize / Threads; // 4

            NS::named_barrier_init<bnum>();

            unsigned int idx = ndi.get_local_id(0);
            unsigned int off = idx * VL * sizeof(int);

            // 2 producers on first iteration, 1 producer on second
            unsigned int indexes[2][2] = {{1, 2},
                                          {3, 3}}; // local ids of producers
            unsigned int prods[2] = {2, 1};        // number of producers

            slm_init(SlmSize * sizeof(int));
            lsc_slm_block_store<int, VL>(off, simd<int, VL>(0));
            barrier();

            for (int b = bnum - 1; b > 0; b--) {
              int j = bnum - b - 1; // iteration index

              bool is_producer = idx == indexes[j][0] || idx == indexes[j][1];
              bool is_consumer = !is_producer;
              // only-consumer or only-producer modes
              unsigned int flag = is_producer ? 0x1 : 0x2;

              unsigned int producers = prods[j];
              unsigned int consumers = Threads - producers;

              if (is_producer) {
                unsigned int p_off = j * sizeof(int) * SlmSize / 4;
                // second iteration store partialy overlaps first iteration
                // stores
                unsigned int dx = producers == 2 ? (idx - 1) : 0;
                p_off += dx * sizeof(int) * SlmSize / 2;
                int v = 0xdead0000 + idx;
                simd<int, SlmSize / 2> init(v);
                // producer stores to SLM
                lsc_slm_block_store<int, SlmSize / 2>(p_off, init);
              }

              NS::named_barrier_signal(b, flag, producers, consumers);

              if (is_consumer)
                NS::named_barrier_wait(b); // consumers waiting for signal

              auto val = lsc_slm_block_load<int, VL>(off); // reading SLM
              // and storing it to output surface
              fence();
              lsc_block_store<int, VL>(acc, off + j * SlmSize * sizeof(int),
                                       val);
              fence();
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0002;
    if (i < Size / 4)
      etalon = 0xdead0001;
    if (i >= Size / 2) {
      if (i < (7 * Size / 8)) {
        if (i < (5 * Size / 8))
          etalon = 0xdead0001;
        else
          etalon = 0xdead0003;
      }
    }
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

  passed &= test<1, 16>(q);
  passed &= test<2, 32>(q);
  passed &= test<3, 64>(q);
  passed &= test<4, 128>(q);
  passed &= test<5, 256>(q);

  return passed ? 0 : 1;
}
