//==------------ multiple_wg.cpp - DPC++ ESIMD on-device test -------------===//
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
// Test checks support of named barrier in ESIMD kernel.
// Basic case with 2 work-groups.
// SLM and surface size is 16 bytes, 8 bytes per group.
// Each work-group contain 2 threads: 1 producer and 1 consumer.
// Producers store to SLM; consumers read SLM and store data to surface.

#include "../esimd_test_utils.hpp"

#define NS __ESIMD_NS

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num> class KernelID;

template <unsigned case_num, unsigned Groups, unsigned Threads, class QueueTY>
bool test(QueueTY q) {
  constexpr unsigned NUM = Threads * Groups;
  constexpr unsigned Size = 4 * NUM;

  static_assert(Threads > 1, "Threads number must be greater than 1");
  static_assert(Threads % 2 == 0, "Threads number expect to be even");
  static_assert(Groups > 1, "Threads number must be greater than 1");

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

            constexpr unsigned producers = Threads / 2;
            constexpr unsigned consumers = Threads / 2;

            // total number of execution units
            constexpr unsigned NUM = Threads * Groups;
            // number of ints per execution unit
            constexpr unsigned VL = Size / NUM;
            // number of ints each producer stored / each consumer loaded
            constexpr unsigned VL2 = 2 * VL;

            NS::named_barrier_init<bnum>();

            unsigned int localID = ndi.get_local_id(0);
            unsigned int groupID = ndi.get_group(0);
            unsigned int globalID = ndi.get_global_id(0);
            unsigned int groupSize = ndi.get_local_range(0);
            unsigned int group_off = VL * groupID * groupSize * sizeof(int);
            unsigned int global_off = VL * globalID * sizeof(int);

            slm_init(Size * sizeof(int));
            slm_block_store(global_off, simd<int, VL>(0));
            barrier();

            // thread with odd local id is producer in each work-group
            bool is_producer = localID % 2 == 1;
            bool is_consumer = !is_producer;
            // only-producer or only-comsumer modes
            unsigned int flag = is_producer ? 0x1 : 0x2;

            if (is_producer) {
              int v = 0xdead0000 | (groupID << 8) | localID;
              // offset inside work-group
              unsigned int off = (localID - 1) * VL * sizeof(int);
              // producer stores data to SLM
              slm_block_store(group_off + off, simd<int, VL2>(v));
            }

            // signaling after data stored
            NS::named_barrier_signal(bid, flag, producers, consumers);

            if (is_consumer) {
              NS::named_barrier_wait(
                  bid); // consumers waiting here for signal from producer
              // offset inside work-group
              unsigned int off = localID * VL * sizeof(int);
              // read SLM and store to output
              auto ret = slm_block_load<int, VL2>(group_off + off);
              lsc_block_store<int, VL2>(acc, group_off + off, ret);
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool passed = true;
  constexpr unsigned elem_per_group = Size / Groups;
  constexpr unsigned elem_per_thread = elem_per_group / Threads;

  for (int i = 0; i < Size; i++) {
    int etalon = 0xdead0000 | (i / elem_per_group) << 8;
    etalon |= (i % elem_per_group) / elem_per_thread;
    if (etalon % 2 == 0)
      etalon += 1;

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

  passed &= test<1, 2, 2>(q);
  passed &= test<2, 2, 4>(q);
  passed &= test<3, 4, 4>(q);
  passed &= test<4, 2, 32>(q);

  return passed ? 0 : 1;
}
