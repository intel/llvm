//==---------------- lsc_block_store.hpp - DPC++ ESIMD on-device test ------==//
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

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel::experimental::esimd::detail;

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          int BlockWidth, int BlockHeight = 1,
          int N = get_lsc_block_2d_data_size<T, 1u, BlockHeight, BlockWidth,
                                             false, false>(),
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
bool test(unsigned SurfaceWidth, unsigned SurfaceHeight, unsigned SurfacePitch,
          int X, int Y) {
  static_assert(BlockWidth > 0, "Block width must be positive");
  static_assert(BlockHeight > 0, "Block height must be positive");
  static_assert((sizeof(T) * BlockWidth) % 4 == 0,
                "Block width must be aligned by DW");
  static_assert(sizeof(T) * BlockWidth <= 64,
                "Block width must be 64B or less");
  static_assert(BlockHeight <= 8, "Block height must be 8 or less");

  T old_val = get_rand<T>();
  T new_val = get_rand<T>();

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  cl::sycl::range<1> GlobalRange{Groups};
  // threads in each group
  cl::sycl::range<1> LocalRange{Threads};
  cl::sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  unsigned SurfaceSize = SurfacePitch * SurfaceHeight;
  unsigned Size = SurfaceSize * Groups * Threads;

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t off = globalID * SurfaceSize;

            simd<T, N> vals(new_val + off, 1);
            // IUT
            lsc_store2d<T, BlockWidth, BlockHeight, L1H, L3H>(
                out + off, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                SurfacePitch * sizeof(T) - 1, X, Y, vals);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    return false;
  }

  bool passed = true;
  for (int gid = 0; gid < Groups * Threads; gid++) {
    T val = new_val + gid * SurfaceSize;

    for (int j = 0; j < SurfaceHeight; j++) {
      for (int i = 0; i < SurfacePitch; i++) {
        T e = old_val;
        // check if inside block
        if ((i >= X) && (i < X + BlockWidth) && (i < SurfaceWidth) &&
            (j >= Y) && (j < Y + BlockHeight))
          e = val++;

        // index in linear buffer
        int idx = i + j * SurfacePitch + gid * SurfaceSize;
        if (out[idx] != e) {
          passed = false;
          std::cout << "out[" << idx << "] = 0x" << std::hex
                    << (uint64_t)out[idx] << " vs etalon = 0x" << (uint64_t)e
                    << std::dec << std::endl;
        }
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);

  return passed;
}
