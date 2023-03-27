//==---------------- lsc_block_load.hpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::detail;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel::experimental::esimd::detail;

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          bool use_prefetch = false>
bool test(unsigned SurfaceWidth, unsigned SurfaceHeight, unsigned SurfacePitch,
          int X, int Y) {
  // Some restrictions based on documentation
  static_assert(!(Transposed && Transformed),
                "Transposed and transformed is not supported");
  static_assert(BlockWidth > 0, "Block width must be positive");
  static_assert(BlockHeight > 0, "Block height must be positive");

  if constexpr (Transposed) {
    static_assert(NBlocks == 1, "Transposed expected to be 1 block only");
    static_assert(sizeof(T) >= 4, "Transposed can only use D32 and D64");
    if constexpr (sizeof(T) == 4) {
      static_assert(BlockWidth <= 8,
                    "D32 transposed allow only block width 8 and less");
      static_assert(BlockHeight <= 32,
                    "D32 transposed allow only block height 32 and less");
    }
    if constexpr (sizeof(T) == 8) {
      static_assert(BlockWidth == 1 || BlockWidth == 2 || BlockWidth == 4,
                    "D64 transposed allow only block width 1/2/4");
      static_assert(BlockHeight == 8,
                    "D64 transposed allow only block height 8");
    }
  } else if constexpr (Transformed) {
    static_assert(sizeof(T) <= 2, "Transformed can only use D8 and D16");
    if constexpr (sizeof(T) == 2 && NBlocks == 4) {
      static_assert(BlockWidth <= 8,
                    "Transformed D16x4 allow only block width 8 and less");
    }
    static_assert((sizeof(T) * BlockWidth) % 4 == 0,
                  "Transformed block width must be aligned by DW");
    static_assert(BlockWidth <= 16,
                  "Transformed block width must be 16 and less");
    static_assert(BlockWidth >= (4 / sizeof(T)),
                  "Minimal transformed block width depends on data size");
    static_assert(BlockHeight <= 32,
                  "Transformed block height must be 32 and less");
    static_assert(BlockHeight >= (4 / sizeof(T)),
                  "Minimal transformed block height depends on data size");
  } else {
    static_assert((sizeof(T) * BlockWidth) % 4 == 0,
                  "Block width must be aligned by DW");
    static_assert(sizeof(T) * BlockWidth * NBlocks <= 64,
                  "Total block width must be 64B or less");
    static_assert(BlockHeight <= 32, "Block height must be 32 or less");
    if constexpr (sizeof(T) == 4) {
      static_assert(NBlocks < 4, "D32 restricted to use 1 or 2 blocks only");
    }
    if constexpr (sizeof(T) == 8) {
      static_assert(NBlocks < 2, "D64 restricted to use 1 block only");
    }
  }

  constexpr int N =
      get_lsc_block_2d_data_size<T, NBlocks, BlockHeight, BlockWidth,
                                 Transposed, Transformed>();
  /* Due to store_2d a is subject to stricter restrictions:
   *   NBlocks always 1, no Transposed, no Transformed, max BlockHeight 8.
   * Series of 2d stores with height 1 are used to write loaded data to output
   * buffer. Also Transformed load_2d extends BlockWidth to the next power of 2
   * and rounds up BlockHeight.
   */
  constexpr int SH = Transformed
                         ? roundUpNextMultiple<BlockHeight, 4 / sizeof(T)>()
                         : BlockHeight;
  constexpr int SW = Transformed ? getNextPowerOf2<BlockWidth>() : BlockWidth;
  constexpr int SN = get_lsc_block_2d_data_size<T, 1u, 1u, SW, false, false>();

  std::cout << "N  = " << N << std::endl;
  std::cout << "SN = " << SN << std::endl;
  std::cout << "W  = " << BlockWidth << " SW = " << SW << std::endl;
  std::cout << "H  = " << BlockHeight << " SH = " << SH << std::endl;

  T old_val = get_rand<T>();

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<sycl::info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  sycl::range<1> GlobalRange{Groups};
  // threads in each group
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  unsigned SurfaceSize = SurfacePitch * SurfaceHeight * NBlocks;
  unsigned Size = SurfaceSize * Groups * Threads;

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  T *in = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    in[i] = get_rand<T>();

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t off = globalID * SurfaceSize;

            unsigned width = SurfaceWidth * sizeof(T) - 1;
            unsigned height = SurfaceHeight - 1;
            unsigned pitch = SurfacePitch * sizeof(T) - 1;

            simd<T, N> vals;
            if constexpr (use_prefetch) {
              lsc_prefetch_2d<T, BlockWidth, BlockHeight, NBlocks, L1H, L3H>(
                  in + off, width, height, pitch, X, Y);
              vals = lsc_load_2d<T, BlockWidth, BlockHeight, NBlocks,
                                 Transposed, Transformed>(in + off, width,
                                                          height, pitch, X, Y);
            } else {
              vals = lsc_load_2d<T, BlockWidth, BlockHeight, NBlocks,
                                 Transposed, Transformed, L1H, L3H>(
                  in + off, width, height, pitch, X, Y);
            }

            for (int i = 0; i < NBlocks; i++) {
              for (int j = 0; j < SH; j++) {
                simd<T, SN> v =
                    vals.template select<SN, 1>(i * SN * SH + j * SW);
                lsc_store_2d<T, SW>(
                    out + off, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                    SurfacePitch * sizeof(T) - 1, X + i * SW, Y + j, v);
              }
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    sycl::free(in, ctx);
    return false;
  }

  bool passed = true;

  if constexpr (Transposed) {
    for (int gid = 0; gid < Groups * Threads; gid++) {
      int dx = 0, dy = 0;
      for (int j = 0; j < SurfaceHeight; j++) {
        for (int i = 0; i < SurfacePitch; i++) {
          T e = old_val;
          // index in linear buffer
          int idx = i + j * SurfacePitch + gid * SurfaceSize;

          // check if inside block
          if ((i >= X) && (i < X + BlockWidth) && (j >= Y) &&
              (j < Y + BlockHeight)) {
            if (i < SurfaceWidth) {
              if (X + dx < SurfaceWidth)
                e = in[(X + dx) + (Y + dy) * SurfacePitch + gid * SurfaceSize];
              else
                e = (T)0;
            }
            dy += 1;
            if (dy == BlockHeight) {
              dy = 0;
              dx += 1;
            }
          }

          if (out[idx] != e) {
            passed = false;
            std::cout << "out" << idx << "] = 0x" << std::hex
                      << (uint64_t)out[idx] << " vs etalon = 0x" << (uint64_t)e
                      << std::dec << std::endl;
          }
        }
      }
    }
  } else if constexpr (Transformed) {
    constexpr int scale = 4 / sizeof(T);
    for (int gid = 0; gid < Groups * Threads; gid++) {
      for (int j = 0; j < SurfaceHeight; j++) {
        for (int i = 0; i < SurfacePitch; i++) {
          T e = old_val;
          // index in linear buffer
          int idx = i + j * SurfacePitch + gid * SurfaceSize;

          // check if inside block
          if ((i >= X) && (i < X + SW * NBlocks) && (j >= Y) && (j < Y + SH)) {
            int di = i - X;
            int dj = j - Y;
            int bn = di / SW;

            int dx, dy;
            dx = di / scale + bn * (BlockWidth - SW / scale) +
                 (dj % scale) * SW / scale;
            dy = dj + di % scale - dj % scale;

            if (i < SurfaceWidth) {
              if (dx < BlockWidth * (bn + 1) && (dx + X) < SurfaceWidth &&
                  (dy + Y) < SurfaceHeight)
                e = in[(X + dx) + (Y + dy) * SurfacePitch + gid * SurfaceSize];
              else
                e = (T)0;
            }
          }

          if (out[idx] != e) {
            passed = false;
            std::cout << std::hex << "out[0x" << idx << "] = 0x"
                      << (uint64_t)out[idx] << " vs etalon = 0x" << (uint64_t)e
                      << std::dec << std::endl;
          }
        }
      }
    }
  } else {
    for (int gid = 0; gid < Groups * Threads; gid++) {
      for (int j = 0; j < SurfaceHeight; j++) {
        for (int i = 0; i < SurfacePitch; i++) {
          T e = old_val;
          // index in linear buffer
          int idx = i + j * SurfacePitch + gid * SurfaceSize;

          // check if inside block
          if ((i >= X) && (i < X + BlockWidth * NBlocks) &&
              (i < SurfaceWidth) && (j >= Y) && (j < Y + BlockHeight))
            e = in[idx];

          if (out[idx] != e) {
            passed = false;
            std::cout << "out[" << idx << "] = 0x" << std::hex
                      << (uint64_t)out[idx] << " vs etalon = 0x" << (uint64_t)e
                      << std::dec << std::endl;
          }
        }
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);
  sycl::free(in, ctx);

  return passed;
}
