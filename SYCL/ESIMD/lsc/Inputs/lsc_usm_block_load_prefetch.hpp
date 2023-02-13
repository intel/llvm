//==-- lsc_usm_block_load_prefetch.hpp - DPC++ ESIMD on-device test --------==//
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

template <typename T, uint16_t N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          bool UsePrefetch = false, bool UseOldValuesOperand = true>
bool test(uint32_t Groups, uint32_t Threads) {
  static_assert(DS != lsc_data_size::u8u32 && DS != lsc_data_size::u16u32,
                "unsupported DS for lsc_block_load()");
  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");

  uint32_t Size = Groups * Threads * N;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;

  auto Q = queue{gpu_selector_v};
  auto D = Q.get_device();
  std::cout << "Running on " << D.get_info<sycl::info::device::name>()
            << ", T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UsePrefetch=" << UsePrefetch
            << ",UseOldValuesOperand=" << UseOldValuesOperand;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *Out = sycl::malloc_shared<T>(Size, Q);
  T *In = sycl::malloc_shared<T>(Size, Q);
  for (int i = 0; i < Size; i++) {
    In[i] = get_rand<T>();
    Out[i] = 0;
  }

  try {
    Q.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
       uint16_t GlobalID = NDI.get_global_id(0);
       uint32_t ElemOffset = GlobalID * N;

       simd<T, N> Vals;
       if constexpr (UseOldValuesOperand) {
         // TODO: these 2 lines work-around the problem with scalar conversions
         // to bfloat16. It could be just: "simd<T, N> OldValues(ElemOffset,
         // 1);"
         simd<uint32_t, N> OldValuesInt(ElemOffset, 1);
         simd<T, N> OldValues = OldValuesInt;

         simd_mask<1> Mask = GlobalID % 1;
         if constexpr (UsePrefetch) {
           lsc_prefetch<T, N, DS, L1H, L3H>(In + ElemOffset);
           Vals = lsc_block_load<T, N, DS>(In + ElemOffset, Mask, OldValues);
         } else {
           Vals = lsc_block_load<T, N, DS, L1H, L3H>(In + ElemOffset, Mask,
                                                     OldValues);
         }
       } else {
         if constexpr (UsePrefetch) {
           lsc_prefetch<T, N, DS, L1H, L3H>(In + ElemOffset);
           Vals = lsc_block_load<T, N, DS>(In + ElemOffset);
         } else {
           Vals = lsc_block_load<T, N, DS, L1H, L3H>(In + ElemOffset);
         }
       }
       lsc_block_store(Out + ElemOffset, Vals);
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    sycl::free(In, Q);
    return false;
  }

  int NumErrors = 0;
  for (int i = 0; i < Size && NumErrors < 32; i++) {
    bool IsMaskSet = (i / N) % 1;
    Tuint Expected = sycl::bit_cast<Tuint>(In[i]);
    Tuint Computed = sycl::bit_cast<Tuint>(Out[i]);

    if (!IsMaskSet) {
      // Values loaded by lsc_block_load() are undefined - skip the check.
      if (!UseOldValuesOperand)
        continue;
      Expected = sycl::bit_cast<Tuint>((T)i);
    }

    if (Computed != Expected) {
      NumErrors++;
      std::cout << "out[" << i << "] = 0x" << std::hex << Computed
                << " vs etalon = 0x" << Expected << std::dec << std::endl;
    }
  }

  std::cout << (NumErrors ? " FAILED" : " passed") << std::endl;
  sycl::free(Out, Q);
  sycl::free(In, Q);
  return NumErrors == 0;
}

template <typename T> bool test_lsc_block_load() {
  constexpr lsc_data_size DS = lsc_data_size::default_size;
  constexpr cache_hint L1H = cache_hint::none;
  constexpr cache_hint L3H = cache_hint::none;

  constexpr bool NoPrefetch = false;
  constexpr bool CheckMerge = true;
  constexpr bool NoCheckMerge = false;

  bool Passed = true;
  Passed &= test<T, 64, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(1, 4);
  Passed &= test<T, 32, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(1, 4);
  Passed &= test<T, 16, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(2, 2);
  Passed &= test<T, 8, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(2, 8);
  Passed &= test<T, 4, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(3, 3);
  if constexpr (sizeof(T) * 2 >= sizeof(int))
    Passed &= test<T, 2, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(5, 5);
  if constexpr (sizeof(T) >= sizeof(int))
    Passed &= test<T, 1, DS, L1H, L3H, NoPrefetch, NoCheckMerge>(3, 5);

  Passed &= test<T, 64, DS, L1H, L3H, NoPrefetch, CheckMerge>(1, 4);
  Passed &= test<T, 32, DS, L1H, L3H, NoPrefetch, CheckMerge>(2, 2);
  Passed &= test<T, 16, DS, L1H, L3H, NoPrefetch, CheckMerge>(4, 4);
  Passed &= test<T, 8, DS, L1H, L3H, NoPrefetch, CheckMerge>(2, 8);
  Passed &= test<T, 4, DS, L1H, L3H, NoPrefetch, CheckMerge>(3, 3);
  if constexpr (sizeof(T) * 2 >= sizeof(int))
    Passed &= test<T, 2, DS, L1H, L3H, NoPrefetch, CheckMerge>(5, 5);
  if constexpr (sizeof(T) >= sizeof(int))
    Passed &= test<T, 1, DS, L1H, L3H, NoPrefetch, CheckMerge>(3, 5);

  return Passed;
}

template <typename T, lsc_data_size DS = lsc_data_size::default_size>
bool test_lsc_prefetch() {
  constexpr cache_hint L1H = cache_hint::cached;
  constexpr cache_hint L3H = cache_hint::uncached;
  constexpr bool DoPrefetch = true;

  bool Passed = true;
  Passed &= test<T, 64, DS, L1H, L3H, DoPrefetch>(1, 4);
  Passed &= test<T, 32, DS, L1H, L3H, DoPrefetch>(1, 4);
  Passed &= test<T, 16, DS, L1H, L3H, DoPrefetch>(2, 2);
  Passed &= test<T, 8, DS, L1H, L3H, DoPrefetch>(2, 8);
  Passed &= test<T, 4, DS, L1H, L3H, DoPrefetch>(3, 3);
  if constexpr (sizeof(T) * 2 >= sizeof(int))
    Passed &= test<T, 2, DS, L1H, L3H, DoPrefetch>(5, 5);
  if constexpr (sizeof(T) >= sizeof(int))
    Passed &= test<T, 1, DS, L1H, L3H, DoPrefetch>(3, 5);

  return Passed;
}
