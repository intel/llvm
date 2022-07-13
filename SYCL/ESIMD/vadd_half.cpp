//==---------------- vadd_half.cpp  - DPC++ ESIMD on-device test
//------------==//
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

// Test addition of simd<sycl::half, N> objects.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
using TstT = half;
using SrcT = half;

template <int N>
using int_type_t = std::conditional_t<
    N == 1, int8_t,
    std::conditional_t<
        N == 2, int16_t,
        std::conditional_t<N == 4, int32_t,
                           std::conditional_t<N == 8, int64_t, void>>>>;

template <class T> auto cast(T val) { return val; }
template <> auto cast<char>(char val) { return (int)val; }
template <> auto cast<unsigned char>(unsigned char val) {
  return (unsigned int)val;
}
#ifdef __SYCL_DEVICE_ONLY__
template <> auto cast<_Float16>(_Float16 val) { return (float)val; }
#endif

int main(int argc, char **argv) {
  constexpr unsigned VL = 32;
  constexpr unsigned Size = 1 * VL;
  // 2048 is a boundary, where arith operation rounding starts to differ between
  // GPU h/w implementation and host emulation - as a result, results may differ
  // by 1 bit.
  // TODO use Gen's control register to adjust behavior.
  int start_val = 2048 - VL / 2;

  if (argc > 1) {
    start_val = atoi(argv[1]);
  }
  std::cout << "Using start value = " << start_val << "\n";

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  TstT *A = malloc_shared<TstT>(Size, q);
  SrcT *B = malloc_shared<SrcT>(Size, q);
  using DstT = __ESIMD_DNS::computation_type_t<TstT, SrcT>;
  std::cout << "Computed dst type = " << typeid(DstT).name() << "\n";
  DstT *C = malloc_shared<DstT>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = start_val + i;
    B[i] = 1;
    C[i] = 0;
  }

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<class Test>([=]() SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        simd<TstT, VL> va(A, vector_aligned_tag{});
        simd<SrcT, VL> vb(B, vector_aligned_tag{});
        simd<DstT, VL> vc = va + vb;
        vc.copy_to(C, vector_aligned_tag{});
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    free(C, q);
    return 1;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    DstT Gold = A[i] + B[i];
    DstT Res = C[i];
    using Tint = int_type_t<sizeof(DstT)>;
    Tint ResBits = *(Tint *)&Res;
    Tint GoldBits = *(Tint *)&Gold;

    if (abs(ResBits - GoldBits) > 1) {
      if (++err_cnt < 100) {
        std::cout << "failed at index " << i << ": " << cast(Res) << "(0x"
                  << std::hex << ResBits << ")"
                  << " != " << std::dec << cast(Gold) << "(0x" << std::hex
                  << GoldBits << std::dec << ")\n";
      }
    } else {
      const char *eq = ResBits == GoldBits ? " == " : " ~ ";
      std::cout << "index " << i << ": " << cast(Res) << "(0x" << std::hex
                << ResBits << ")" << eq << std::dec << cast(Gold) << "(0x"
                << std::hex << GoldBits << std::dec << ")\n";
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }
  free(A, q);
  free(B, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
