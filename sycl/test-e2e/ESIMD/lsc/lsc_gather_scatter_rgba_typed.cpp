//==--- lsc_gather_scatter_rgba_typed.cpp - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The LSC typed-surface "quad" messages are supported on Xe2 and later.
// REQUIRES: aspect-ext_intel_legacy_image
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_lnl_m || arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h || arch-intel_gpu_wcl || arch-intel_gpu_nvl_u || arch-intel_gpu_nvl_s
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The test checks the functionality of the Xe2+ typed-surface (image) RGBA LSC
// gather and scatter ESIMD APIs: lsc_gather_rgba_typed / lsc_scatter_rgba_typed
// (and exercises lsc_prefetch_rgba_typed). A kernel reads the pixels of an
// input image with lsc_gather_rgba_typed, adds a per-channel constant and
// stores the result into an output image with lsc_scatter_rgba_typed. A second
// gather masks off the odd pixels and checks that their channels are taken from
// the pass-through argument. The results are then verified on the host.

#include "../esimd_test_utils.hpp"

#include <sycl/accessor_image.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iexp = sycl::ext::intel::experimental::esimd;

// Image dimensions (in pixels) and SIMD width.
static constexpr unsigned Width = 256;
static constexpr unsigned Height = 8;
static constexpr unsigned N = 16;

// Per-channel constants added by the kernel.
static constexpr uint32_t DR = 1, DG = 2, DB = 3, DA = 4;

// Pass-through value of channel C of SIMD lane I.
static constexpr uint32_t passThru(unsigned C, unsigned I) {
  return 0xF0000000u | (C << 8) | I;
}

int main() {
  // 4 channels per pixel.
  std::vector<uint32_t> InBuf(Width * Height * 4);
  std::vector<uint32_t> OutBuf(Width * Height * 4, 0);
  std::vector<uint32_t> MergeBuf(Width * Height * 4, 0);

  for (unsigned y = 0; y < Height; ++y) {
    for (unsigned x = 0; x < Width; ++x) {
      unsigned Pixel = y * Width + x;
      InBuf[Pixel * 4 + 0] = Pixel;       // R
      InBuf[Pixel * 4 + 1] = Pixel + 100; // G
      InBuf[Pixel * 4 + 2] = Pixel + 200; // B
      InBuf[Pixel * 4 + 3] = Pixel + 300; // A
    }
  }

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  try {
    image<2> ImgIn(InBuf.data(), image_channel_order::rgba,
                   image_channel_type::unsigned_int32, range<2>{Width, Height});
    image<2> ImgOut(OutBuf.data(), image_channel_order::rgba,
                    image_channel_type::unsigned_int32,
                    range<2>{Width, Height});
    image<2> ImgMerge(MergeBuf.data(), image_channel_order::rgba,
                      image_channel_type::unsigned_int32,
                      range<2>{Width, Height});

    range<2> GlobalRange{Width / N, Height};

    q.submit([&](handler &cgh) {
       auto AccIn = ImgIn.get_access<uint4, access::mode::read>(cgh);
       auto AccOut = ImgOut.get_access<uint4, access::mode::write>(cgh);
       auto AccMerge = ImgMerge.get_access<uint4, access::mode::write>(cgh);
       cgh.parallel_for<class LscTypedRGBA>(
           GlobalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
             uint32_t BaseX = it.get_id(0) * N;
             uint32_t Y = it.get_id(1);

             simd<uint32_t, N> U(BaseX, 1);
             simd<uint32_t, N> V = Y;
             simd<uint32_t, N> R = 0;
             simd<uint32_t, N> LOD = 0;

             // Prefetch (exercises lsc_prefetch_rgba_typed; no observable
             // effect on correctness).
             iexp::lsc_prefetch_rgba_typed<uint32_t, N>(AccIn, U, V);

             // Read all 4 channels; result is laid out channel-major.
             simd<uint32_t, N * 4> Px =
                 iexp::lsc_gather_rgba_typed<uint32_t, N>(AccIn, U, V);

             Px.select<N, 1>(0 * N) += DR;
             Px.select<N, 1>(1 * N) += DG;
             Px.select<N, 1>(2 * N) += DB;
             Px.select<N, 1>(3 * N) += DA;

             iexp::lsc_scatter_rgba_typed<uint32_t, N>(AccOut, U, V, R, LOD,
                                                       Px);

             simd<uint32_t, N * 4> PassThru;
             for (unsigned C = 0; C < 4; ++C)
               for (unsigned I = 0; I < N; ++I)
                 PassThru[C * N + I] = passThru(C, I);
             simd_mask<N> EvenX = (U & 1) == 0;
             simd<uint32_t, N * 4> Merged =
                 iexp::lsc_gather_rgba_typed<uint32_t, N>(AccIn, U, V, R, LOD,
                                                          EvenX, PassThru);
             iexp::lsc_scatter_rgba_typed<uint32_t, N>(AccMerge, U, V, R, LOD,
                                                       Merged);
           });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  unsigned NumErrors = 0;
  for (unsigned Pixel = 0; Pixel < Width * Height && NumErrors < 16; ++Pixel) {
    uint32_t Exp[4] = {InBuf[Pixel * 4 + 0] + DR, InBuf[Pixel * 4 + 1] + DG,
                       InBuf[Pixel * 4 + 2] + DB, InBuf[Pixel * 4 + 3] + DA};
    for (int C = 0; C < 4; ++C) {
      uint32_t Got = OutBuf[Pixel * 4 + C];
      if (Got != Exp[C]) {
        std::cout << "Error at pixel " << Pixel << " channel " << C << ": got "
                  << Got << " expected " << Exp[C] << "\n";
        ++NumErrors;
      }
    }
  }

  for (unsigned Pixel = 0; Pixel < Width * Height && NumErrors < 16; ++Pixel) {
    unsigned X = Pixel % Width;
    for (unsigned C = 0; C < 4; ++C) {
      uint32_t Exp = X % 2 == 0 ? InBuf[Pixel * 4 + C] : passThru(C, X % N);
      uint32_t Got = MergeBuf[Pixel * 4 + C];
      if (Got != Exp) {
        std::cout << "Pass-through error at pixel " << Pixel << " channel " << C
                  << ": got " << Got << " expected " << Exp << "\n";
        ++NumErrors;
      }
    }
  }

  std::cout << (NumErrors ? "FAILED\n" : "Passed\n");
  return NumErrors ? 1 : 0;
}
