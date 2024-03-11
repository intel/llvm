//==---------------- vadd_raw_send.cpp  - DPC++ ESIMD on-device test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-gen9
// UNSUPPORTED: gpu-intel-dg1,gpu-intel-dg2,gpu-intel-pvc
// RUN: %{build} -fno-sycl-esimd-force-stateless-mem -o %t1.out
// RUN: %{run} %t1.out
// RUN: %{build} -fno-sycl-esimd-force-stateless-mem -DUSE_CONSTEXPR_API -o %t2.out
// RUN: %{run} %t2.out
// RUN: %{build} -fno-sycl-esimd-force-stateless-mem -DUSE_SUPPORTED_API -o %t3.out
// RUN: %{run} %t3.out
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
#ifdef USE_CONSTEXPR_API
  return experimental::esimd::raw_send<execSize, sfid, numSrc0, numDst>(
      oldDst, src0, exDesc, desc);
#elif defined(USE_SUPPORTED_API)
  return esimd::raw_send<execSize, sfid, numSrc0, numDst>(oldDst, src0, exDesc,
                                                          desc);
#else
  return experimental::esimd::raw_send(oldDst, src0, exDesc, desc, execSize,
                                       sfid, numSrc0, numDst);
#endif
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
#ifdef USE_CONSTEXPR_API
  return experimental::esimd::raw_sends<execSize, sfid, numSrc0, numSrc1>(
      src0, data, exDesc, desc);
#elif defined(USE_SUPPORTED_API)
  return esimd::raw_sends<execSize, sfid, numSrc0, numSrc1>(src0, data, exDesc,
                                                            desc);
#else
  return experimental::esimd::raw_sends(src0, data, exDesc, desc, execSize,
                                        sfid, numSrc0, numSrc1);
#endif
}

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE void block_write2(AccessorTy acc, unsigned int offset,
                               simd<T, N> data) {
  simd<uint32_t, 16> src0;
  auto src0_ref1 =
      src0.template select<8, 1>(0).template bit_cast_view<unsigned int>();
  auto src0_ref2 = src0.template select<8, 1>(8);

  src0_ref1.template select<1, 1>(2) = offset >> 4;
  src0_ref2 = data.template bit_cast_view<uint32_t>();
  uint32_t exDesc = 0xA;
  SurfaceIndex desc = esimd::get_surface_index(acc);
  desc += 0x40A0200;
  constexpr uint8_t execSize = 0x83;
  constexpr uint8_t sfid = 0x0;
  constexpr uint8_t numSrc0 = 0x2;
#ifdef USE_CONSTEXPR_API
  return experimental::esimd::raw_send<execSize, sfid, numSrc0>(src0, exDesc,
                                                                desc);
#elif defined(USE_SUPPORTED_API)
  return esimd::raw_send<execSize, sfid, numSrc0>(src0, exDesc, desc);
#else
  return experimental::esimd::raw_send(src0, exDesc, desc, execSize, sfid,
                                       numSrc0);
#endif
}

template <typename T> int test(queue q) {
  constexpr unsigned Size = 1024 * 128;
  constexpr unsigned VL = sizeof(T) == 4 ? 16 : 32;
  T *A = new T[Size];
  T *B = new T[Size];
  T *C = new T[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0;
  }

  try {
    buffer<T, 1> bufa(A, range<1>(Size));
    buffer<T, 1> bufb(B, range<1>(Size));
    buffer<T, 1> bufc(C, range<1>(Size));

    // We need that many workgroups
    range<1> GlobalRange{Size / VL};

    // We need that many threads in each group
    range<1> LocalRange{1};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.template get_access<access::mode::read>(cgh);
      auto PB = bufb.template get_access<access::mode::read>(cgh);
      auto PC = bufc.template get_access<access::mode::write>(cgh);
      cgh.parallel_for(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned int offset = i * VL * sizeof(T);
            simd<T, VL> va = dwaligned_block_read<T, VL>(PA, offset);
            simd<T, VL> vb = dwaligned_block_read<T, VL>(PB, offset);
            simd<T, VL> vc = va + vb;
            constexpr int SIZE = VL / 2;
            block_write1(PC, offset, vc.template select<SIZE, 1>(0).read());
            offset += SIZE * sizeof(T);
            block_write2(PC, offset, vc.template select<SIZE, 1>(SIZE).read());
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

  delete[] A;
  delete[] B;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt;
}

int main(void) {

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  int err_cnt = 0;

  err_cnt += test<float>(q);
  if (dev.has(sycl::aspect::fp16)) {
    err_cnt += test<sycl::half>(q);
  }
  return err_cnt > 0 ? 1 : 0;
}
