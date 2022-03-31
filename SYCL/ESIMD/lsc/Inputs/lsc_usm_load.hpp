//==------- lsc_flat_load.hpp - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

#include "common.hpp"

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          uint16_t VL, uint16_t VS, bool transpose,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          bool use_prefetch = false>
bool test(uint32_t pmask = 0xffffffff) {
  static_assert((VL == 1) || !transpose, "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!transpose, "Conversion types may not use vector");
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");
  static_assert(sizeof(T) >= 4,
                "D8 and D16 are valid only in 2D block load/store");

  if constexpr (!transpose && VS > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  uint16_t Size = Groups * Threads * VL * VS;

  T vmask = (T)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = (T)0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = (T)0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = (T)0xffff0000;

  T old_val = get_rand<T>();
  T zero_val = (T)0;

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

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  T *in = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    in[i] = get_rand<T>();

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t elem_off = globalID * VL * VS;
            uint32_t byte_off = elem_off * sizeof(T);

            if constexpr (transpose) {
              simd<T, VS> vals;
              if constexpr (use_prefetch) {
                lsc_prefetch<T, VS, DS, L1H, L3H>(in + elem_off);
                vals = lsc_block_load<T, VS, DS>(in + elem_off);
              } else {
                vals = lsc_block_load<T, VS, DS, L1H, L3H>(in + elem_off);
              }
              lsc_block_store<T, VS, lsc_data_size::default_size>(
                  out + elem_off, vals);
            } else {
              simd<uint32_t, VL> offset(byte_off, VS * sizeof(T));
              simd_mask<VL> pred;
              for (int i = 0; i < VL; i++)
                pred.template select<1, 1>(i) = (pmask >> i) & 1;

              simd<T, VS * VL> vals;
              if constexpr (use_prefetch) {
                lsc_prefetch<T, VS, DS, L1H, L3H, VL>(in, offset, pred);
                vals = lsc_gather<T, VS, DS, cache_hint::none, cache_hint::none,
                                  VL>(in, offset, pred);
              } else {
                vals = lsc_gather<T, VS, DS, L1H, L3H, VL>(in, offset, pred);
              }

              if constexpr (DS == lsc_data_size::u8u32 ||
                            DS == lsc_data_size::u16u32)
                vals &= vmask;

              lsc_scatter<T, VS, lsc_data_size::default_size, cache_hint::none,
                          cache_hint::none, VL>(out, offset, vals, pred);
            }
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    sycl::free(in, ctx);
    return false;
  }

  bool passed = true;

  if constexpr (transpose) {
    for (int i = 0; i < Size; i++) {
      T e = in[i];
      if (out[i] != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << (uint64_t)out[i]
                  << " vs etalon = 0x" << (uint64_t)e << std::dec << std::endl;
      }
    }
  } else {
    for (int i = 0; i < Size; i++) {
      T e = (pmask >> ((i / VS) % VL)) & 1 ? in[i] & vmask : old_val;
      if (out[i] != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << (uint64_t)out[i]
                  << " vs etalon = 0x" << (uint64_t)e << std::dec << std::endl;
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);
  sycl::free(in, ctx);

  return passed;
}
