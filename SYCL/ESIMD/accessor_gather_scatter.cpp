//==------- accessor_gather_scatter.cpp  - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the gather/scatter accessor-based ESIMD
// intrinsics.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

template <typename T>
using Acc = accessor<T, 1, access_mode::read_write, access::target::device>;

#define MASKED_LANE_NUM_REV 1

template <typename T, unsigned VL, unsigned STRIDE> struct Kernel {
  Acc<T> acc;
  Kernel(Acc<T> acc) : acc(acc) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::experimental::esimd;
    uint32_t ii = static_cast<uint32_t>(i.get(0));
    // every STRIDE threads (subgroups with sg_size=1) access contiguous block
    // of STRIDE*VL elements
    uint32_t global_offset = (ii / STRIDE) * VL * STRIDE + ii % STRIDE;
    simd<uint32_t, VL> offsets(0, STRIDE);
    simd<T, VL> v =
        gather<T, VL>(acc, offsets * sizeof(T), global_offset * sizeof(T));
    v += ii;
    simd_mask<VL> pred = 1;
    pred.template select<1, 1>(VL - MASKED_LANE_NUM_REV) =
        0; // mask out the last lane
    scatter<T, VL>(acc, offsets * sizeof(T), v, global_offset * sizeof(T),
                   pred);
  }
};

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  size_t size = VL * STRIDE * 117;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << "...\n";
  T *A = new T[size];

  for (unsigned i = 0; i < size; ++i) {
    A[i] = (T)i;
  }

  try {
    buffer<T, 1> buf(A, range<1>(size));
    range<1> glob_range{size / VL};

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      Kernel<T, VL, STRIDE> kernel(acc);
      cgh.parallel_for(glob_range, kernel);
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < size; ++i) {
    T gold = (T)i;
    // the sequential number of sub group block (STRIDE in each) i falls into
    unsigned sg_block_num = i / (VL * STRIDE);
    // the start of the i index block this sg block covers
    unsigned sg_block_start_i = sg_block_num * VL * STRIDE;
    // the local id (within block) of the sg covering this i
    unsigned sg_local_id = (i - sg_block_start_i) % STRIDE;
    // the global id of the sg covering this i
    unsigned sg_global_id = sg_local_id + sg_block_num * STRIDE;

    unsigned lane_id = ((i % (VL * STRIDE)) - sg_local_id) / STRIDE;

    gold += lane_id == VL - MASKED_LANE_NUM_REV ? 0 : sg_global_id;

    if (A[i] != gold) {
      if (++err_cnt < 35) {
        std::cout << "failed at index " << i << ": " << A[i] << " != " << gold
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  delete[] A;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char, 8, 1>(q);
  passed &= test<char, 16, 3>(q);
  passed &= test<short, 8, 8>(q);
  passed &= test<short, 16, 1>(q);
  passed &= test<int, 8, 2>(q);
  passed &= test<int, 16, 1>(q);
  passed &= test<float, 8, 2>(q);
  passed &= test<float, 16, 1>(q);
  return passed ? 0 : 1;
}
