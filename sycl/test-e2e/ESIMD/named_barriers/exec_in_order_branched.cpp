//==------- exec_in_order_branched.cpp - DPC++ ESIMD on-device test -------===//
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
// Threads are executed in ascending order of their local ID and each thread
// stores data to addresses that partially overlap with addresses used by
// previous thread. Same as "exec_in_order.cpp", but each thread in separate
// 'if' branch.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num> class KernelID;

template <int case_num, unsigned Size, bool UseSLM, class QueueTY>
bool test(QueueTY q) {
  constexpr unsigned Groups = 1;
  constexpr unsigned Threads = 4;
  // number of ints stored by each thread
  constexpr unsigned VL = Size / Threads;

  static_assert(
      Size % (2 * Threads) == 0,
      "Surface size must be evenly divisible by twice the number of threads");

  // need to write at least 2 ints per thread in order to overlap
  static_assert(VL >= 2,
                "Surface size must be at least 2 times the number of threads");

  std::cout << "Case #" << case_num << "\n\tTreads: " << Threads
            << "\n\tInts per thread: " << VL
            << "\n\tMemory: " << (UseSLM ? "local\n" : "global\n");

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
            // Threads - 1 named barriers required
            // but id 0 reserved for unnamed
            named_barrier_init<Threads>();

            unsigned int idx = ndi.get_local_id(0);
            // overlaping offset
            unsigned int off = idx * VL * sizeof(int) / 2;

            int flag = 0; // producer-consumer mode
            int producers = 2;
            int consumers = 2;

            simd<int, VL> val(idx);

            if constexpr (UseSLM) {
              slm_init(Size * sizeof(int));
              lsc_slm_block_store<int, VL>(2 * off, simd<int, VL>(0));
            }

            barrier();

            // Threads are executed in ascending order of their local ID and
            // each thread stores data to addresses that partially overlap with
            // addresses used by previous thread.
            if (idx == 0) {
              if constexpr (UseSLM) {
                lsc_slm_block_store<int, VL>(off, val);
              } else {
                lsc_fence();
                lsc_block_store<int, VL>(acc, off, val);
                lsc_fence();
              }

              // T0 signals barrier 1 and locks
              // waiting for first signal from T1
              const int barrier_id = 1;
              named_barrier_signal(barrier_id, flag, producers, consumers);
              named_barrier_wait(barrier_id);
            } else if (idx == 1) {
              // T1 signals barrier 1 and locks, waiting for signal from T0
              const int barrier_id = 1;
              named_barrier_signal(barrier_id, flag, producers, consumers);
              named_barrier_wait(barrier_id);

              if constexpr (UseSLM) {
                lsc_slm_block_store<int, VL>(off, val);
              } else {
                lsc_fence();
                lsc_block_store<int, VL>(acc, off, val);
                lsc_fence();
              }

              // T1 signals barrier 2 and locks
              // waiting for first signal from T2
              const int barrier_id2 = 2;
              named_barrier_signal(barrier_id2, flag, producers, consumers);
              named_barrier_wait(barrier_id2);
            } else if (idx == 2) {
              // T2 signals barrier 2 and locks
              // waiting for second signal from T1
              const int barrier_id = 2;
              named_barrier_signal(barrier_id, flag, producers, consumers);
              named_barrier_wait(barrier_id);

              if constexpr (UseSLM) {
                lsc_slm_block_store<int, VL>(off, val);
              } else {
                lsc_fence();
                lsc_block_store<int, VL>(acc, off, val);
                lsc_fence();
              }

              // T2 signals barrier 3 and locks, waiting for signal from T3
              const int barrier_id2 = 3;
              named_barrier_signal(barrier_id2, flag, producers, consumers);
              named_barrier_wait(barrier_id2);
            } else {
              // T3 signals barrier 3 and locks
              // waiting for second signal from T2
              const int barrier_id = 3;
              named_barrier_signal(barrier_id, flag, producers, consumers);
              named_barrier_wait(barrier_id);

              if constexpr (UseSLM) {
                lsc_slm_block_store<int, VL>(off, val);
              } else {
                lsc_fence();
                lsc_block_store<int, VL>(acc, off, val);
                lsc_fence();
              }
            }

            barrier();
            if constexpr (UseSLM) {
              auto res = lsc_slm_block_load<int, VL>(2 * off);
              lsc_block_store<int, VL>(acc, 2 * off, res);
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
    int etalon = i * 2 * Threads / Size;
    if (etalon == Threads) // last stored chunk
      etalon -= 1;
    if (etalon > Threads) // excessive part of surface
      etalon = 0;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << out[i] << " vs " << etalon << "\n";
    }
  }

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}

int main() {
  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;

  // Threads == 4
  passed &= test<1, 8, true>(q);
  passed &= test<2, 8, false>(q);

  passed &= test<3, 16, true>(q);
  passed &= test<4, 16, false>(q);

  passed &= test<5, 32, true>(q);
  passed &= test<6, 32, false>(q);

  passed &= test<7, 64, true>(q);
  passed &= test<8, 64, false>(q);

  passed &= test<9, 128, true>(q);
  passed &= test<10, 128, false>(q);

  return passed ? 0 : 1;
}
