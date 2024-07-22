//==---------------- loop.cpp - DPC++ ESIMD on-device test ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Test checks support of named barrier in a loop in ESIMD kernel.
// SLM and surface size is 32 bytes, 16 bytes per iteration.
// Each iteration has 1 barrier and 1 producer. Producer stores data to SLM,
// then all threads read SLM and store data to surface.

#include "../esimd_test_utils.hpp"

#define NS __ESIMD_NS

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

            constexpr unsigned producers = 1;
            constexpr unsigned consumers = Threads; // 8

            // SLM size is half of output surface size so
            // content of SLM can be copied to out buffer on each iteration
            constexpr unsigned SlmSize = Size / 2;
            // number of ints read/written by single thread
            constexpr unsigned VL = SlmSize / Threads;

            NS::named_barrier_init<bnum>();

            unsigned int idx = ndi.get_local_id(0);
            unsigned int off = idx * VL * sizeof(int);

            slm_init(SlmSize * sizeof(int));
            lsc_slm_block_store<int, VL>(off, simd<int, VL>(0));
            barrier();

            for (int b = 1; b < bnum; b++) {
              int i = b - 1;

              // local ID 1 is producer on first iteration, local ID 2 on second
              bool is_producer = idx == b;
              bool is_consumer = !is_producer;
              // producer is also a consumer
              unsigned int flag = is_producer ? 0x0 : 0x2;

              if (is_producer) {
                // second iteration store partialy overlaps data stored on first
                // iteration
                unsigned int prod_off = i * sizeof(int) * SlmSize / 4;

                int v = 0xdead0000 + idx;
                simd<int, SlmSize / 2> init(v);
                // producer stores to SLM
                lsc_slm_block_store<int, SlmSize / 2>(prod_off, init);
              }

              NS::named_barrier_signal(b, flag, producers, consumers);
              NS::named_barrier_wait(b); // consumers waiting for signal

              // reading SLM
              auto val = lsc_slm_block_load<int, VL>(off);
              // and storing it to output surface
              unsigned int store_off = off + i * SlmSize * sizeof(int);
              fence();
              lsc_block_store<int, VL>(acc, store_off, val);
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
    int etalon = 0;
    if (i < Size / 4)
      etalon = 0xdead0001;
    if (i >= Size / 2) {
      if (i < (7 * Size / 8)) {
        if (i < (5 * Size / 8))
          etalon = 0xdead0001;
        else
          etalon = 0xdead0002;
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
