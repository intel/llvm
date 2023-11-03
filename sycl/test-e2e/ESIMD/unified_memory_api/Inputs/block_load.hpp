//==-- block_load.hpp - DPC++ ESIMD on-device test -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>

#include "../../esimd_test_utils.hpp"
#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// Returns true iff verification is passed.
template <typename T>
bool verify(const T *In, const T *Out, size_t Size, int N,
            bool UsePassThruOperand) {
  int NumErrors = 0;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;
  for (int i = 0; i < Size && NumErrors < 32; i++) {
    bool IsMaskSet = (i / N + 1) % 1;
    Tuint Expected = sycl::bit_cast<Tuint>(In[i]);
    Tuint Computed = sycl::bit_cast<Tuint>(Out[i]);

    if (!IsMaskSet) {
      if (!UsePassThruOperand)
        continue; // the value is undefined without merge operand

      Expected = sycl::bit_cast<Tuint>((T)i);
    }

    if (Computed != Expected) {
      NumErrors++;
      std::cout << "out[" << i << "] = 0x" << std::hex << Computed
                << " vs etalon = 0x" << Expected << std::dec << std::endl;
    }
  }
  std::cout << (NumErrors == 0 ? " passed\n" : " FAILED\n");
  return NumErrors == 0;
}

template <typename T, uint16_t N, bool UseMask, bool UsePassThruOperand,
          bool CheckProperties, typename LoadPropertiesT>
bool testUSM(queue Q, uint32_t Groups, uint32_t Threads,
             LoadPropertiesT LoadProperties) {
  static_assert(!UsePassThruOperand || UseMask, "Cannot merge without mask");

  uint32_t Size = Groups * Threads * N;
  std::cout << "USM case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask
            << ",UsePassThruOperand=" << UsePassThruOperand;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  constexpr size_t Alignment = getAlignment<T, N, UseMask>(LoadProperties);
  T *In = sycl::aligned_alloc_shared<T>(Alignment, Size, Q);
  T *Out = sycl::aligned_alloc_shared<T>(Alignment, Size, Q);
  for (int i = 0; i < Size; i++) {
    In[i] = esimd_test::getRandomValue<T>();
    Out[i] = 0;
  }

  try {
    Q.submit([&](handler &CGH) {
       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = NDI.get_global_id(0);
         uint32_t ElemOffset = GlobalID * N;

         simd<T, N> Vals;
         simd_mask<1> Mask = (GlobalID + 1) % 1;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
             // TODO: these 2 lines work-around the problem with scalar
             // conversions to bfloat16. It could be just: "simd<T, N>
             // PassThru(ElemOffset, 1);"
             simd<uint32_t, N> PassThruInt(ElemOffset, 1);
             simd<T, N> PassThru = PassThruInt;

             if (GlobalID & 0x1)
               Vals = block_load<T, N>(In + ElemOffset, Mask, PassThru);
             else
               Vals =
                   block_load<T, N>(In, ElemOffset * sizeof(T), Mask, PassThru);
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (GlobalID & 0x1)
                 Vals = block_load<T, N>(In + ElemOffset, Mask);
               else
                 Vals = block_load<T, N>(In, ElemOffset * sizeof(T), Mask);
             } else { // if !UseMask
               if (GlobalID & 0x1)
                 Vals = block_load<T, N>(In + ElemOffset);
               else
                 Vals = block_load<T, N>(In, ElemOffset * sizeof(T));
             }
           }
         } else { // if CheckProperties
           if constexpr (UsePassThruOperand) {
             // TODO: these 2 lines work-around the problem with scalar
             // conversions to bfloat16. It could be just: "simd<T, N>
             // PassThru(ElemOffset, 1);"
             simd<uint32_t, N> PassThruInt(ElemOffset, 1);
             simd<T, N> PassThru = PassThruInt;

             if (GlobalID & 0x1)
               Vals = block_load<T, N>(In + ElemOffset, Mask, PassThru,
                                       LoadPropertiesT{});
             else
               Vals = block_load<T, N>(In, ElemOffset * sizeof(T), Mask,
                                       PassThru, LoadPropertiesT{});
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (GlobalID & 0x1)
                 Vals =
                     block_load<T, N>(In + ElemOffset, Mask, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(In, ElemOffset * sizeof(T), Mask,
                                         LoadPropertiesT{});
             } else { // if !UseMask
               if (GlobalID & 0x1)
                 Vals = block_load<T, N>(In + ElemOffset, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(In, ElemOffset * sizeof(T),
                                         LoadPropertiesT{});
             }
           }
         }
         Vals.copy_to(Out + ElemOffset);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(In, Q);
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify<T>(In, Out, Size, N, UsePassThruOperand);
  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T, bool TestPVCFeatures> bool testUSM(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  properties AlignOnlyProps{alignment<16>};

  bool Passed = true;

  // Test block_load() that is available on Gen12 and PVC.
  Passed &= testUSM<T, 1, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 2, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 1, 4, AlignOnlyProps);
  Passed &= testUSM<T, 3, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 8, AlignOnlyProps);
  Passed &= testUSM<T, 4, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 8, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 16, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 32, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  // Intentionally check non-power-of-2 simd size - it must work.
  Passed &= testUSM<T, 33, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 67, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  // Intentionally check big simd size - it must work.
  Passed &= testUSM<T, 512, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 1024, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);

  // Test block_load() without passing compile-time properties argument.
  Passed &= testUSM<T, 16, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &= testUSM<T, 32, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, 4, AlignOnlyProps);

  if constexpr (TestPVCFeatures) {
    // Using mask or cache hints adds the requirement to run tests on PVC.
    // Also, PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.

    properties PVCProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>, alignment<16>};

    if constexpr (sizeof(T) >= 4) // only d/q words are supported now
      Passed &= testUSM<T, 1, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 2, 4, PVCProps);
    if constexpr (sizeof(T) >= 2) // only d/q words are supported now
      Passed &= testUSM<T, 2, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 4, !CheckMask, !CheckMerge, CheckProperties>(Q, 5, 5,
                                                                      PVCProps);
    Passed &= testUSM<T, 8, !CheckMask, !CheckMerge, CheckProperties>(Q, 5, 5,
                                                                      PVCProps);
    Passed &= testUSM<T, 16, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 5, 5, PVCProps);
    Passed &= testUSM<T, 32, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, 4, PVCProps);
    Passed &= testUSM<T, 64, CheckMask, !CheckMerge, CheckProperties>(Q, 7, 1,
                                                                      PVCProps);
    if constexpr (128 * sizeof(T) <= 512)
      Passed &= testUSM<T, 128, CheckMask, CheckMerge, CheckProperties>(
          Q, 1, 4, PVCProps);
    if constexpr (256 * sizeof(T) <= 512)
      Passed &= testUSM<T, 256, CheckMask, CheckMerge, CheckProperties>(
          Q, 1, 4, PVCProps);
    if constexpr (512 * sizeof(T) <= 512)
      Passed &= testUSM<T, 512, CheckMask, CheckMerge, CheckProperties>(
          Q, 1, 4, PVCProps);
  } // TestPVCFeatures

  return Passed;
}

template <typename T, uint16_t N, bool UseMask, bool UsePassThruOperand,
          bool CheckProperties, typename LoadPropertiesT>
bool testACC(queue Q, uint32_t Groups, uint32_t Threads,
             LoadPropertiesT LoadProperties) {
  using host_allocator = sycl::usm_allocator<T, sycl::usm::alloc::host, 16>;
  using host_vector = std::vector<T, host_allocator>;
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;

  static_assert(!UsePassThruOperand || UseMask, "Cannot merge without mask");

  uint32_t Size = Groups * Threads * N;
  std::cout << "ACC case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask
            << ",UsePassThruOperand=" << UsePassThruOperand;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  constexpr size_t Alignment = getAlignment<T, N, UseMask>(LoadProperties);
  host_vector In(Size, host_allocator{Q});
  shared_vector Out(Size, shared_allocator{Q});
  for (int i = 0; i < Size; i++) {
    In[i] = esimd_test::getRandomValue<T>();
    Out[i] = 0;
  }

  try {
    buffer<T, 1> InBuf(Size);
    Q.submit([&](handler &CGH) {
       accessor InAcc{InBuf, CGH};
       auto OutPtr = Out.data();

       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = NDI.get_global_id(0);
         uint32_t ElemOffset = GlobalID * N;

         simd<T, N> Vals;
         simd_mask<1> Mask = (GlobalID + 1) % 1;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
             // TODO: these 2 lines work-around the problem with scalar
             // conversions to bfloat16. It could be just: "simd<T, N>
             // PassThru(ElemOffset, 1);"
             simd<uint32_t, N> PassThruInt(ElemOffset, 1);
             simd<T, N> PassThru = PassThruInt;

             if (ElemOffset == 0) // try the variant without byte-offset
               Vals = block_load<T, N>(InAcc, Mask, PassThru);
             else
               Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T), Mask,
                                       PassThru);
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (ElemOffset == 0)
                 Vals = block_load<T, N>(InAcc, Mask);
               else
                 Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T), Mask);
             } else { // if !UseMask
               if (ElemOffset == 0)
                 Vals = block_load<T, N>(InAcc);
               else
                 Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T));
             }
           }
         } else { // if CheckProperties
           if constexpr (UsePassThruOperand) {
             // TODO: these 2 lines work-around the problem with scalar
             // conversions to bfloat16. It could be just: "simd<T, N>
             // PassThru(ElemOffset, 1);"
             simd<uint32_t, N> PassThruInt(ElemOffset, 1);
             simd<T, N> PassThru = PassThruInt;

             if (ElemOffset == 0)
               Vals =
                   block_load<T, N>(InAcc, Mask, PassThru, LoadPropertiesT{});
             else
               Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T), Mask,
                                       PassThru, LoadPropertiesT{});
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (ElemOffset == 0)
                 Vals = block_load<T, N>(InAcc, Mask, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T), Mask,
                                         LoadPropertiesT{});
             } else { // if !UseMask
               if (ElemOffset == 0)
                 Vals = block_load<T, N>(InAcc, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(InAcc, ElemOffset * sizeof(T),
                                         LoadPropertiesT{});
             }
           }
         }
         Vals.copy_to(OutPtr + ElemOffset);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed = verify<T>(In.data(), Out.data(), Size, N, UsePassThruOperand);
  return Passed;
}

template <typename T, bool TestPVCFeatures> bool testACC(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  properties AlignOnlyProps{alignment<16>};

  bool Passed = true;

  // Test block_load() that is available on Gen12 and PVC:
  // 1, 2, 4 or 8  16-byte loads.
  constexpr int NElemsInOword = 16 / sizeof(T);
  Passed &= testACC<T, NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignOnlyProps);
  Passed &=
      testACC<T, 2 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 1, 4, AlignOnlyProps);
  Passed &=
      testACC<T, 4 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 2, 4, AlignOnlyProps);
  Passed &=
      testACC<T, 8 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 2, 4, AlignOnlyProps);

  // Test block_load() without passing compile-time properties argument.
  Passed &=
      testACC<T, NElemsInOword, !CheckMask, !CheckMerge, !CheckProperties>(
          Q, 2, 4, AlignOnlyProps);

  if constexpr (TestPVCFeatures) {
    // Using mask or cache hints adds the requirement to run tests on PVC.
    // Also, PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.

    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    properties PVCProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>, alignment<16>};

    // Test block_load() that is available on Gen12 and PVC:
    // 1, 2, 4 or 8  16-byte loads
    Passed &=
        testACC<T, 1 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 2 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 1, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 3 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 8, AlignOnlyProps);
    Passed &=
        testACC<T, 4 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 8 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 16 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 32 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);
    Passed &=
        testACC<T, 64 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, AlignOnlyProps);

    if constexpr (sizeof(T) <= 4)
      Passed &=
          testACC<T, 128 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
              Q, 2, 4, AlignOnlyProps);
  } // TestPVCFeatures

  return Passed;
}
