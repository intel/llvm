//==---------------- dpas_test3.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -DESIMD_XE_HPC %s -DVER1 -o %t.out1
// RUN: %clangxx -fsycl -DESIMD_XE_HPC %s -DVER2 -o %t.out2
// RUN: %GPU_RUN_PLACEHOLDER %t.out1
// RUN: %GPU_RUN_PLACEHOLDER %t.out2

// The test verifies the low-level API for DPAS with 'bfloat16' types.
// The macros VER1 and VER2 are used to verify slightly different
// ways of initializing the input operands of DPAS. There were runtime
// errors previously depending on what variant of initialization was used.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using BF16 = uint16_t;

union BFloat16 {
  float f;
  unsigned short s[2];
};

uint16_t FP32toBF16(float f) {
  BFloat16 bf16;
  bf16.f = f;
  return bf16.s[1];
}

float BF16toFP32(uint16_t i) {
  BFloat16 bf16;
  bf16.s[0] = 0;
  bf16.s[1] = i;
  return bf16.f;
}

template <typename T, int K, int N>
simd<T, K * N> pack_bb(simd<T, K * N> &src) {
  // K=16 N=16
  simd<T, K * N> dst;
  auto dst2d = dst.template bit_cast_view<T, K / 2, N * 2>();
  auto src2d = src.template bit_cast_view<T, K, N>();
  dst2d.template select<8, 1, 16, 2>(0, 0) =
      src2d.template select<8, 2, 16, 1>(0, 0);
  dst2d.template select<8, 1, 16, 2>(0, 1) =
      src2d.template select<8, 2, 16, 1>(1, 0);
  return dst;
}

void dpas_ker(nd_item<1> &idx, BF16 *matA, BF16 *matB, float *matC) {
  //  matC = matC + matA * matB
  //  matC 8x16 MxN
  //  matA 8x16 MxK
  //  matB 16x16 KxN
  constexpr int MB = 8;
  constexpr int NB = 16; // KB = NB = 16 in pvc
  constexpr int KB = 16;
  constexpr int TN = 128;
  constexpr int TN1 = 128;
  constexpr int TN2 = 64;
  constexpr int REPEAT_COUNT = 8;
  constexpr int SYSTOLIC_DEPTH = 8;

  simd<BF16, MB * KB> BA;  // MB, KB
  simd<BF16, KB * NB> BB;  // KB, NB
  simd<float, MB * NB> BC; // MB, NB
#ifdef VER1
  BA.copy_from(matA);
  BB.copy_from(matB);
#else // VER2
  for (int i = 0; i < MB * KB; ++i)
    BA[i] = FP32toBF16(float(i));
  for (int i = 0; i < KB * NB; ++i)
    BB[i] = FP32toBF16(float(i));
#endif
  BC = 0.0f;
  simd<BF16, KB *NB> BBvnni = pack_bb<BF16, KB, NB>(BB);
  BC = dpas<argument_type::BF16, argument_type::BF16, SYSTOLIC_DEPTH,
            REPEAT_COUNT, float, uint, uint, TN, TN1, TN2>(
      BC, BBvnni.template bit_cast_view<uint>(),
      BA.template bit_cast_view<uint>());
  BBvnni.copy_to(matB);
  BC.copy_to(matC);
}

int main() {
  // A [8][16] * B[16][16]= C[8][16]
  queue q(gpu_selector{});
  nd_range<1> Range(range<1>{1}, range<1>{1});
  constexpr int MB = 8;
  constexpr int NB = 16; // KB = NB = 16 in pvc
  constexpr int KB = 16;
  BF16 *matA = malloc_shared<BF16>(MB * KB, q);
  BF16 *matB = malloc_shared<BF16>(KB * NB, q);
  float *matC = malloc_shared<float>(MB * NB, q);
  for (int i = 0; i < MB * KB; ++i)
    matA[i] = FP32toBF16(float(i));
  for (int i = 0; i < KB * NB; ++i)
    matB[i] = FP32toBF16(float(i));
  for (int i = 0; i < MB * NB; ++i)
    matC[i] = 0.0f;
  q.submit([&](handler &cgh) {
     cgh.parallel_for(Range, [=](nd_item<1> idx) SYCL_ESIMD_KERNEL {
       dpas_ker(idx, matA, matB, matC);
     });
   }).wait();

  unsigned err_cnt = 0;
  for (unsigned i = 0; i < MB * NB && err_cnt < 10; ++i) {
    int m = i / NB;
    int n = i % NB;
    float res = 0.0f;
    for (int k = 0; k < KB; ++k)
      res += float((m * KB + k) * (k * NB + n));
    if (std::abs(res - matC[i]) > 0.0001) {
      std::cerr << "res vs ref: " << res << " : " << matC[i] << std::endl;
      err_cnt++;
    }
  }
  free(matA, q);
  free(matB, q);
  free(matC, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
