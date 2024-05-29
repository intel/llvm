//==---------------- lsc_store_2d.hpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>

#include <iostream>

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <unsigned int N, unsigned int M>
constexpr unsigned int roundUpNextMultiple() {
  return ((N + M - 1) / M) * M;
}

/// Compute next power of 2 of a constexpr with guaranteed compile-time
/// evaluation.
template <unsigned int N, unsigned int K, bool K_gt_eq_N> struct NextPowerOf2;
template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, true> {
  static constexpr unsigned int get() { return K; }
};
template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, false> {
  static constexpr unsigned int get() {
    return NextPowerOf2<N, K * 2, K * 2 >= N>::get();
  }
};

template <unsigned int N> constexpr unsigned int getNextPowerOf2() {
  return NextPowerOf2<N, 1, (1 >= N)>::get();
}
template <> constexpr unsigned int getNextPowerOf2<0>() { return 0; }

// Compute the data size for 2d block load or store.
template <typename T, int NBlocks, int Height, int Width, bool Transposed,
          bool Transformed>
constexpr int get_lsc_block_2d_data_size() {
  if (Transformed)
    return roundUpNextMultiple<Height, 4 / sizeof(T)>() *
           getNextPowerOf2<Width>() * NBlocks;
  return Width * Height * NBlocks;
}

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          int BlockWidth, int BlockHeight = 1,
          int N = get_lsc_block_2d_data_size<T, 1u, BlockHeight, BlockWidth,
                                             false, false>(),
          cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
bool test(unsigned SurfaceWidth, unsigned SurfaceHeight, unsigned SurfacePitch,
          int X, int Y) {

  T old_val = get_rand<T>();
  T new_val = get_rand<T>();

  auto GPUSelector = gpu_selector_v;
  auto q = queue{GPUSelector};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<sycl::info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  sycl::range<1> GlobalRange{Groups};
  // threads in each group
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  unsigned SurfaceSize = SurfacePitch * SurfaceHeight;
  unsigned Size = SurfaceSize * Groups * Threads;

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t off = globalID * SurfaceSize;

            simd<T, N> vals(new_val + off, 1);
            // IUT
            lsc_store_2d<T, BlockWidth, BlockHeight, L1H, L2H>(
                out + off, SurfaceWidth * sizeof(T) - 1, SurfaceHeight - 1,
                SurfacePitch * sizeof(T) - 1, X, Y, vals);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
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
