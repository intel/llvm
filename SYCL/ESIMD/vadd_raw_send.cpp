//==---------------- vadd_raw_send.cpp  - DPC++ ESIMD on-device test
//-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-gen9
// UNSUPPORTED: gpu-intel-dg1,gpu-intel-dg2,cuda,hip
// TODO: esimd_emulator fails due to unimplemented 'raw_send' intrinsic
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// The test checks raw send functionality with block read/write implementation
// on SKL. It does not work on DG1 due to send instruction incompatibility.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE simd<T, N> dwaligned_block_read(AccessorTy acc,
                                             unsigned int offset) {
  simd<unsigned int, 8> src0;
  simd<T, N> oldDst;

  src0.select<1, 1>(2) = offset;
  uint32_t exDesc = 0xA;
  SurfaceIndex desc = esimd::get_surface_index(acc);
  desc += 0x2284300;
  constexpr uint8_t execSize = 0x84;
  constexpr uint8_t sfid = 0x0;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numDst = 0x2;

  return experimental::esimd::raw_send_load(oldDst, src0, exDesc, desc,
                                            execSize, sfid, numSrc0, numDst);
}

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE void block_write1(AccessorTy acc, unsigned int offset,
                               simd<T, N> data) {
  simd<unsigned int, 8> src0;

  src0.template select<1, 1>(2) = offset >> 4;
  uint32_t exDesc = 0x4A;
  SurfaceIndex desc = esimd::get_surface_index(acc);
  desc += 0x20A0200;
  constexpr uint8_t execSize = 0x83;
  constexpr uint8_t sfid = 0x0;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numSrc1 = 0x1;

  return experimental::esimd::raw_sends_store(src0, data, exDesc, desc,
                                              execSize, sfid, numSrc0, numSrc1);
}

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE void block_write2(AccessorTy acc, unsigned int offset,
                               simd<T, N> data) {
  simd<T, 16> src0;
  auto src0_ref1 =
      src0.template select<8, 1>(0).template bit_cast_view<unsigned int>();
  auto src0_ref2 = src0.template select<8, 1>(8);

  src0_ref1.template select<1, 1>(2) = offset >> 4;
  src0_ref2 = data;
  uint32_t exDesc = 0xA;
  SurfaceIndex desc = esimd::get_surface_index(acc);
  desc += 0x40A0200;
  constexpr uint8_t execSize = 0x83;
  constexpr uint8_t sfid = 0x0;
  constexpr uint8_t numSrc0 = 0x2;

  return experimental::esimd::raw_send_store(src0, exDesc, desc, execSize, sfid,
                                             numSrc0);
}

int main(void) {
  constexpr unsigned Size = 1024 * 128;
  constexpr unsigned VL = 16;

  float *A = new float[Size];
  float *B = new float[Size];
  float *C = new float[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  try {
    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));

    // We need that many workgroups
    range<1> GlobalRange{Size / VL};

    // We need that many threads in each group
    range<1> LocalRange{1};

    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va = dwaligned_block_read<float, VL>(PA, offset);
            simd<float, VL> vb = dwaligned_block_read<float, VL>(PB, offset);
            simd<float, VL> vc = va + vb;
            constexpr int SIZE = VL / 2;
            block_write1(PC, offset, vc.select<SIZE, 1>(0).read());
            offset += SIZE * sizeof(float);
            block_write2(PC, offset, vc.select<SIZE, 1>(SIZE).read());
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    delete[] A;
    delete[] B;
    delete[] C;
    return 1;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  delete[] A;
  delete[] B;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
