//==---------------- vadd_2d.cpp  - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 8;
  constexpr unsigned GroupSize = 2;

  int A[Size];
  int B[Size];
  int C[Size] = {};

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  {
    cl::sycl::image<2> imgA(A, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});
    cl::sycl::image<2> imgB(B, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});
    cl::sycl::image<2> imgC(C, image_channel_order::rgba,
                            image_channel_type::unsigned_int32,
                            range<2>{Size / 4, 1});

    // We need that many workitems
    cl::sycl::range<1> GlobalRange{(Size / VL)};

    // Number of workitems in a workgroup
    cl::sycl::range<1> LocalRange{GroupSize};

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](cl::sycl::handler &cgh) {
      auto accA = imgA.get_access<uint4, cl::sycl::access::mode::read>(cgh);
      auto accB = imgB.get_access<uint4, cl::sycl::access::mode::read>(cgh);
      auto accC = imgC.get_access<uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;

            constexpr int ESIZE = sizeof(int);
            int x = i * ESIZE * VL;
            int y = 0;

            simd<int, VL> va;
            auto va_ref = va.format<int, 1, VL>();
            va_ref = media_block_load<int, 1, VL>(accA, x, y);

            simd<int, VL> vb;
            auto vb_ref = vb.format<int, 1, VL>();
            vb_ref = media_block_load<int, 1, VL>(accB, x, y);

            simd<int, VL> vc;
            auto vc_ref = vc.format<int, 1, VL>();
            vc_ref = va_ref + vb_ref;
            media_block_store<int, 1, VL>(accC, x, y, vc_ref);
          });
    });
    e.wait();
  }

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                << " + " << B[i] << "\n";
      return 1;
    }
  }

  std::cout << "Passed\n";
  return 0;
}
