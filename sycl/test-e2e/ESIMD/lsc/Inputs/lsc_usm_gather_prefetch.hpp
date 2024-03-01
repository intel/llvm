//==------- lsc_usm_gather_prefetch.hpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

#include "../../esimd_test_utils.hpp"
#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t Toffset;
#else
typedef uint32_t Toffset;
#endif

template <typename T, uint16_t VL, uint16_t VS,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none,
          bool use_prefetch = false, bool use_old_values = false>
bool test(queue q, uint32_t Groups, uint32_t Threads,
          uint32_t pmask = 0xffffffff) {
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }
  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");

  if constexpr (VS > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  std::cout << "Running case: T=" << esimd_test::type_name<T>()
            << ", Groups=" << Groups << ", Threads=" << Threads << ", VL=" << VL
            << ", VS=" << VS << ", DS=" << esimd_test::toString(DS)
            << ", use_prefetch=" << use_prefetch
            << ", use_old_values=" << use_old_values << std::endl;

  uint16_t Size = Groups * Threads * VL * VS;
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;
  Tuint vmask = (Tuint)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = 0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = 0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = 0xffff0000;

  T old_val = get_rand<T>();
  T zero_val = (T)0;
  T merge_value = old_val * 2;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), q));
  for (int i = 0; i < Size; i++)
    out[i] = old_val;

  T *in = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), q));
  for (int i = 0; i < Size; i++)
    in[i] = get_rand<T>();

  try {
    q.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
       uint16_t globalID = ndi.get_global_id(0);
       uint32_t elem_off = globalID * VL * VS;
       uint32_t byte_off = elem_off * sizeof(T);

#ifndef USE_SCALAR_OFFSET
       simd<Toffset, VL> offset(byte_off, VS * sizeof(T));
#else
        Toffset offset = byte_off;
#endif
       simd_mask<VL> pred;
       for (int i = 0; i < VL; i++)
         pred.template select<1, 1>(i) = (pmask >> i) & 1;

       simd<T, VS * VL> vals;
       if constexpr (use_prefetch) {
         lsc_prefetch<T, VS, DS, L1H, L2H, VL>(in, offset, pred);
         vals = lsc_gather<T, VS, DS, cache_hint::none, cache_hint::none, VL>(
             in, offset, pred);
       } else if constexpr (!use_old_values) {
         vals = lsc_gather<T, VS, DS, L1H, L2H, VL>(in, offset, pred);
       } else { // use_old_values
         simd<T, VS *VL> old_values = merge_value;
         vals =
             lsc_gather<T, VS, DS, L1H, L2H, VL>(in, offset, pred, old_values);
       }

       if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32)
         vals &= vmask;

       if constexpr (use_old_values) {
         // Verifying usage of old_values operand in gather requires storing
         // the whole loaded vector.
         lsc_scatter<T, VS>(out, offset, vals);
       } else {
         lsc_scatter<T, VS>(out, offset, vals, pred);
       }
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, q);
    sycl::free(in, q);
    return false;
  }

  int num_errors = 0;

  if constexpr (use_old_values)
    old_val = merge_value;
  for (int i = 0; i < Size; i++) {
    Tuint in_val = sycl::bit_cast<Tuint>(in[i]);
    Tuint out_val = sycl::bit_cast<Tuint>(out[i]);
#ifndef USE_SCALAR_OFFSET
    Tuint e = (pmask >> ((i / VS) % VL)) & 1 ? in_val
                                             : sycl::bit_cast<Tuint>(old_val);
#else
    // Calculate the mask to identify the areas that were actually updated
    constexpr uint16_t mask =
        1U << ((sycl::bit_cast<uint32_t>((float)VL) >> 23) - 126);
    Tuint e = ((i / VS) % VL == 0) && (pmask >> ((i / VS) % VL)) & (mask - 1)
                  ? in_val
                  : sycl::bit_cast<Tuint>(old_val);
#endif
    e &= vmask;
    out_val &= vmask;
    if (out_val != e && num_errors++ < 32) {
      std::cout << "Error: out[" << i << "] = 0x" << std::hex << out_val
                << " vs etalon = 0x" << e << std::dec << std::endl;
    }
  }

  if (num_errors)
    std::cout << "Case FAILED" << std::endl;

  sycl::free(out, q);
  sycl::free(in, q);

  return num_errors == 0;
}

template <typename T, lsc_data_size DS = lsc_data_size::default_size,
          bool DoPrefetch>
bool test_lsc_gather_prefetch(queue q) {
  constexpr cache_hint L1H = cache_hint::cached;
  constexpr cache_hint L2H = cache_hint::uncached;
  constexpr bool DoMerging = true;

  bool Passed = true;
  Passed &=
      test<T, 1, 1, DS, L1H, L2H, DoPrefetch, !DoMerging>(q, 4, 4, rand());
  if constexpr (!DoPrefetch)
    Passed &=
        test<T, 1, 1, DS, L1H, L2H, DoPrefetch, DoMerging>(q, 4, 4, rand());

#ifndef USE_SCALAR_OFFSET
  // These tests use lsc_scatter with scalar offset when USE_SCALAR_OFFSET macro
  // is set, which is UB and thus guarded by the macro here.
  Passed &= test<T, 32, 1, DS, L1H, L2H, DoPrefetch>(q, 1, 4, rand());
  Passed &= test<T, 16, 1, DS, L1H, L2H, DoPrefetch>(q, 2, 4, rand());
  Passed &= test<T, 8, 1, DS, L1H, L2H, DoPrefetch>(q, 2, 2, rand());
  Passed &= test<T, 4, 1, DS, L1H, L2H, DoPrefetch>(q, 4, 2, rand());
  Passed &= test<T, 2, 1, DS, L1H, L2H, DoPrefetch>(q, 4, 16, rand());

  // The next block of tests is only for gather with merging semantics,
  // not for prefetch tests.
  if constexpr (!DoPrefetch) {
    Passed &=
        test<T, 32, 1, DS, L1H, L2H, DoPrefetch, DoMerging>(q, 1, 4, rand());
    Passed &=
        test<T, 2, 1, DS, L1H, L2H, DoPrefetch, DoMerging>(q, 4, 16, rand());
  }

  if constexpr (((DS == lsc_data_size::default_size && sizeof(T) >= 4) ||
                 DS == lsc_data_size::u32 || DS == lsc_data_size::u32) &&
                !DoPrefetch) {
    Passed &= test<T, 32, 2, DS, L1H, L2H, DoPrefetch>(q, 2, 4, rand());
  }
#endif // !USE_SCALAR_OFFSET

  return Passed;
}

template <typename T, lsc_data_size DS = lsc_data_size::default_size>
bool test_lsc_gather() {
  auto q = queue{gpu_selector_v};
  std::cout << "Running lsc_gather() tests for T=" << esimd_test::type_name<T>()
            << " on " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  constexpr bool NoPrefetch = false;
  return test_lsc_gather_prefetch<T, DS, NoPrefetch>(q);
}

template <typename T, lsc_data_size DS = lsc_data_size::default_size,
          bool IsGatherLikePrefetch = true>
std::enable_if_t<IsGatherLikePrefetch, bool> test_lsc_prefetch() {
  auto q = queue{gpu_selector_v};
  std::cout << "Running gather-like lsc_prefetch() tests for T="
            << esimd_test::type_name<T>() << " on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  constexpr bool DoPrefetch = true;
  return test_lsc_gather_prefetch<T, DS, DoPrefetch>(q);
}
