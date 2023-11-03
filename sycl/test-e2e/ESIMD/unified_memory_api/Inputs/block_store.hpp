//==------- block_store.hpp - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T, uint16_t N, bool UseMask, bool UseProperties,
          typename StorePropertiesT>
bool testUSM(queue Q, uint32_t Groups, uint32_t Threads,
             StorePropertiesT StoreProperties) {

  uint16_t Size = Groups * Threads * N;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;

  std::cout << "USM case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask << ",UseProperties=" << UseProperties
            << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};
  constexpr size_t Alignment = getAlignment<T, N, UseMask>(StoreProperties);
  T *Out = sycl::aligned_alloc_shared<T>(Alignment, Size, Q);
  T Out_val = esimd_test::getRandomValue<T>();
  for (int i = 0; i < Size; i++)
    Out[i] = Out_val;

  try {
    Q.submit([&](handler &cgh) {
       cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = ndi.get_global_id(0);
         uint32_t ElemOff = GlobalID * N;
         //  TODO: these 2 lines work-around the problem with scalar
         //  conversions to bfloat16. It could be just: "simd<T, N>
         //  PassThru(ElemOffset, 1);"
         simd<uint32_t, N> PassThruInt(ElemOff, 1);
         simd<T, N> Vals = PassThruInt;
         if constexpr (UseMask) {
           simd_mask<1> Mask = (GlobalID + 1) & 0x1;
           block_store(Out + ElemOff, Vals, Mask, StorePropertiesT{});
           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 1;
           block_store(Out, ElemOff * sizeof(T), Vals, Mask,
                       StorePropertiesT{});
           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 2;
           auto View = Vals.template select<N, 1>();
           block_store<T, N>(Out, ElemOff * sizeof(T), View, Mask,
                             StorePropertiesT{});
           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 3;
           View = Vals.template select<N, 1>();
           block_store<T, N>(Out + ElemOff, View, Mask, StorePropertiesT{});
         } else {
           if constexpr (UseProperties)
             block_store(Out + ElemOff, Vals, StorePropertiesT{});

           else
             block_store(Out + ElemOff, Vals);

           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 1;
           if constexpr (UseProperties)
             block_store(Out, ElemOff * sizeof(T), Vals, StorePropertiesT{});
           else
             block_store(Out, ElemOff * sizeof(T), Vals);

           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 2;
           auto View = Vals.template select<N, 1>();
           if constexpr (UseProperties)
             block_store<T, N>(Out, ElemOff * sizeof(T), View,
                               StorePropertiesT{});
           else
             block_store<T, N>(Out, ElemOff * sizeof(T), View);

           Vals = block_load<T, N>(Out + ElemOff);
           Vals += 3;
           View = Vals.template select<N, 1>();
           if constexpr (UseProperties)
             block_store<T, N>(Out + ElemOff, View, StorePropertiesT{});
           else
             block_store<T, N>(Out + ElemOff, View);
         }
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = true;

  for (int i = 0; i < Size; i++) {
    bool IsMaskSet = (i / N + 1) & 0x1;
    Tuint Expected = sycl::bit_cast<Tuint>(Out_val);
    if (!UseMask || IsMaskSet)
      Expected = sycl::bit_cast<Tuint>((T)(i + 6));
    Tuint Computed = sycl::bit_cast<Tuint>(Out[i]);
    if (Computed != Expected) {
      Passed = false;
      std::cout << "Out[" << i << "] = " << std::to_string(Computed) << " vs "
                << std::to_string(Expected) << std::endl;
    }
  }

  sycl::free(Out, Q);

  return Passed;
}

template <typename T, bool TestPVCFeatures> bool test_block_store(queue Q) {
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;
  properties AlignOnlyProps{alignment<sizeof(T)>};

  bool Passed = true;

  // Test block_store() that is available on Gen12 and PVC.
  Passed &= testUSM<T, 1, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 2, !CheckMask, CheckProperties>(Q, 1, 4, AlignOnlyProps);
  Passed &= testUSM<T, 3, !CheckMask, CheckProperties>(Q, 2, 8, AlignOnlyProps);
  Passed &= testUSM<T, 4, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 8, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &=
      testUSM<T, 16, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &=
      testUSM<T, 32, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  // Intentionally check non-power-of-2 simd size - it must work.
  Passed &=
      testUSM<T, 33, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  // TODO: Enable after failure fixed
  // Passed &=
  //    testUSM<T, 67, !CheckMask, CheckProperties>(Q, 1, 4, AlignOnlyProps);
  // Intentionally check big simd size - it must work.
  Passed &=
      testUSM<T, 128, !CheckMask, CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &=
      testUSM<T, 256, !CheckMask, CheckProperties>(Q, 1, 4, AlignOnlyProps);

  // Test block_store() without passing compile-time properties argument.
  Passed &=
      testUSM<T, 16, !CheckMask, !CheckProperties>(Q, 2, 4, AlignOnlyProps);
  Passed &=
      testUSM<T, 32, !CheckMask, !CheckProperties>(Q, 2, 4, AlignOnlyProps);

  if constexpr (TestPVCFeatures) {
    // Using cache hints adds the requirement to run tests on PVC.
    // Also, PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.
    properties PVCProps{cache_hint_L1<cache_hint::write_back>,
                        cache_hint_L2<cache_hint::write_back>, alignment<16>};

    if constexpr (sizeof(T) >= 4) // only d/q words are supported now
      Passed &= testUSM<T, 1, !CheckMask, CheckProperties>(Q, 2, 4, PVCProps);
    if constexpr (sizeof(T) >= 2) // only d/q words are supported now
      Passed &= testUSM<T, 2, !CheckMask, CheckProperties>(Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 4, !CheckMask, CheckProperties>(Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 8, !CheckMask, CheckProperties>(Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 16, CheckMask, CheckProperties>(Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 32, !CheckMask, CheckProperties>(Q, 2, 4, PVCProps);
    Passed &= testUSM<T, 64, !CheckMask, CheckProperties>(Q, 7, 1, PVCProps);
    if constexpr (128 * sizeof(T) <= 512)
      Passed &= testUSM<T, 128, CheckMask, CheckProperties>(Q, 1, 4, PVCProps);
    if constexpr (256 * sizeof(T) <= 512)
      Passed &= testUSM<T, 256, CheckMask, CheckProperties>(Q, 1, 4, PVCProps);
  } // TestPVCFeatures

  return Passed;
}
