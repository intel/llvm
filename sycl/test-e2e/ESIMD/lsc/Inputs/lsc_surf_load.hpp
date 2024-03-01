//==------- lsc_surf_load.hpp - DPC++ ESIMD on-device test -----------------==//
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
using namespace sycl::ext::intel::experimental::esimd;

template <
    int case_num, typename T, uint32_t Groups, uint32_t Threads, uint16_t VL,
    uint16_t VS, bool transpose, lsc_data_size DS = lsc_data_size::default_size,
    cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none,
    bool use_prefetch = false, typename Flags = __ESIMD_NS::overaligned_tag<4>>
bool test(uint32_t pmask = 0xffffffff) {
  static_assert((VL == 1) || !transpose, "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!transpose, "Conversion types may not use vector");
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");

  if constexpr (!transpose && VS > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  uint16_t Size = Groups * Threads * VL * VS;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;
  Tuint vmask = (Tuint)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = (T)0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = (T)0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = (T)0xffff0000;

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

  using aligned_allocator =
      sycl::usm_allocator<T, sycl::usm::alloc::shared,
                          Flags::template alignment<__ESIMD_DNS::__raw_t<T>>>;
  aligned_allocator Allocator(q);

  std::vector<T, aligned_allocator> out(Size, old_val, Allocator);
  std::vector<T, aligned_allocator> in(Size, Allocator);
  for (int i = 0; i < Size; i++)
    in[i] = get_rand<T>();

  try {
    buffer<T, 1> bufo(out.data(), out.size());
    buffer<T, 1> bufi(in.data(), in.size());

    auto e = q.submit([&](handler &cgh) {
      auto acco = bufo.template get_access<access::mode::write>(cgh);
      auto acci = bufi.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint16_t globalID = ndi.get_global_id(0);
            uint32_t elem_off = globalID * VL * VS;
            uint32_t byte_off = elem_off * sizeof(T);

            if constexpr (transpose) {
              simd<T, VS> vals;
              if constexpr (use_prefetch) {
                lsc_prefetch<T, VS, DS, L1H, L2H>(acci, byte_off);
                if constexpr (sizeof(T) < 8) {
                  vals = lsc_block_load<T, VS, DS, L1H, L2H>(acci, byte_off,
                                                             Flags{});
                } else {
                  vals = lsc_block_load<T, VS, DS, L1H, L2H>(acci, byte_off);
                }
              } else {
                if constexpr (sizeof(T) < 8) {
                  vals = lsc_block_load<T, VS, DS, L1H, L2H>(acci, byte_off,
                                                             Flags{});
                } else {
                  vals = lsc_block_load<T, VS, DS, L1H, L2H>(acci, byte_off);
                }
              }
              if constexpr (sizeof(T) < 8) {
                lsc_block_store<T, VS, DS>(acco, byte_off, vals, Flags{});
              } else {
                lsc_block_store<T, VS, DS>(acco, byte_off, vals);
              }
            } else {
              simd<uint32_t, VL> offset(byte_off, VS * sizeof(T));
              simd_mask<VL> pred;
              for (int i = 0; i < VL; i++)
                pred.template select<1, 1>(i) = (pmask >> i) & 1;

              simd<T, VS * VL> vals;
              if constexpr (use_prefetch) {
                lsc_prefetch<T, VS, DS, L1H, L2H, VL>(acci, offset, pred);
                vals = lsc_gather<T, VS, DS, cache_hint::none, cache_hint::none,
                                  VL>(acci, offset, pred);
              } else {
                vals = lsc_gather<T, VS, DS, L1H, L2H, VL>(acci, offset, pred);
              }

              if constexpr (DS == lsc_data_size::u8u32 ||
                            DS == lsc_data_size::u16u32)
                vals &= vmask;

              lsc_scatter<T, VS, lsc_data_size::default_size, cache_hint::none,
                          cache_hint::none, VL>(acco, offset, vals, pred);
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool passed = true;

  if constexpr (transpose) {
    for (int i = 0; i < Size; i++) {
      Tuint e = sycl::bit_cast<Tuint>(in[i]);
      Tuint out_val = sycl::bit_cast<Tuint>(out[i]);
      if (out_val != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << out_val
                  << " vs etalon = 0x" << e << std::dec << std::endl;
      }
    }
  } else {
    for (int i = 0; i < Size; i++) {
      Tuint in_val = sycl::bit_cast<Tuint>(in[i]);
      Tuint out_val = sycl::bit_cast<Tuint>(out[i]);
      Tuint e = (pmask >> ((i / VS) % VL)) & 1 ? in_val & vmask
                                               : sycl::bit_cast<Tuint>(old_val);
      if (out_val != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << out_val
                  << " vs etalon = 0x" << e << std::dec << std::endl;
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  return passed;
}
