//==------- variable_gather_mask.cpp  - DPC++ ESIMD on-device test ---------==//
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
// This is a regression test for the VC BE bug which generates incorrect code in
// some cases in presence of variable (not compile-time constant) mask
// (aka predicate) in the scatter operation.

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

using Acc = accessor<int, 1, access_mode::read_write, access::target::device>;

constexpr int VL = 8;
constexpr int MASKED_LANE = 4;
static int MaskedLane = MASKED_LANE;

struct KernelAcc {
  int masked_lane;
  Acc acc;

  KernelAcc(Acc acc) : acc(acc), masked_lane(MaskedLane) {}

  void operator()() const SYCL_ESIMD_KERNEL {
#ifdef CONST_MASK
    constexpr int masked_lane = MASKED_LANE;
#endif
#ifdef USE_RAW_API
    constexpr int val = MASKED_LANE;
    simd<int, VL>::raw_vector_type v{val, val, val, val, val, val, val, val};
    simd_mask<VL>::raw_vector_type pred{1, 1, 1, 1, 1, 1, 1, 1};
    pred[masked_lane] = 0;
#if defined(USE_WORKAROUND)
    pred = __builtin_convertvector(pred == 1, simd_mask<VL>::raw_vector_type);
#endif
    simd<uint32_t, VL>::raw_vector_type offsets = {0, 4, 8, 12, 16, 20, 24, 28};
    SurfaceIndex si = get_surface_index(acc);
    __esimd_scatter_scaled<int, VL, unsigned int, 2, 0>(pred, si, 0, offsets,
                                                        v);
#else
    simd<int, VL> v(MASKED_LANE);
    simd_mask<VL> pred = 1;
    pred[masked_lane] = 0;
    // Workaround for the BE bug:
#if defined(USE_WORKAROUND)
    pred = pred == 1;
#endif
    simd<uint32_t, VL> offsets(0, sizeof(int));
    scatter(acc, offsets, v, 0, pred);
#endif
  }
};

struct KernelUSM {
  int masked_lane;
  int *ptr;

  KernelUSM(int *ptr) : ptr(ptr), masked_lane(MaskedLane) {}

  void operator()() const SYCL_ESIMD_KERNEL {
#ifdef CONST_MASK
    constexpr int masked_lane = MASKED_LANE;
#endif
    simd<int, VL> v(MASKED_LANE);
    simd_mask<VL> pred = 1;
    pred[masked_lane] = 0;
    // Workaround for the BE bug:
#if defined(USE_WORKAROUND)
    pred = pred == 1;
#endif
    simd<uint32_t, VL> offsets(0, sizeof(int));
    scatter(ptr, offsets, v, pred);
  }
};

template <class Command> bool test(const char *msg, queue q, Command cmd) {
  std::cout << "Testing " << msg << "...\n";
  constexpr int size = VL;
  int *B = malloc_shared<int>(size, q); // works for accessor too

  for (int i = 0; i < size; ++i) {
    B[i] = (int)-1;
  }

  try {
    auto e = cmd(B);
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(B, q);
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < size; ++i) {
    auto gold = i == MaskedLane ? -1 : MaskedLane;

    if (B[i] != gold) {
      if (err_cnt++ < VL) {
        std::cout << "failed at index " << i << ": " << B[i] << " != " << gold
                  << " (gold)\n";
      }
    }
  }
  sycl::free(B, q);

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  } else {
    std::cout << "  OK\n";
  }
  return err_cnt == 0;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  std::cout << "MaskedLane=" << MASKED_LANE << "\n";

  bool passed = true;

  passed &= test("accessor", q, [&](int *B) {
    sycl::buffer<int, 1> Bbuf(B, range<1>(VL));

    return q.submit([&](handler &cgh) {
      auto b = Bbuf.template get_access<access::mode::read_write>(cgh);
      KernelAcc kernel(b);
      cgh.single_task(kernel);
    });
  });

  passed &= test("USM", q, [&](int *B) {
    return q.submit([&](handler &cgh) {
      KernelUSM kernel(B);
      cgh.single_task(kernel);
    });
  });

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
