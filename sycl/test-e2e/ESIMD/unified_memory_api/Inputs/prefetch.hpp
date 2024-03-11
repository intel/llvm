//==------------- prefetch.hpp - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains tests that use various variants of prefetch API to make
// sure they do not cause any issues.

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T>
bool verify(const T *In, const T *Out, int N, int Size, int VS) {
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;
  const Tuint *InI = reinterpret_cast<const Tuint *>(In);
  const Tuint *OutI = reinterpret_cast<const Tuint *>(Out);

  int NOffsets = N / VS;
  int NumErrors = 0;

  for (uint32_t I = 0; I < Size; I += N) { // Verify by 1 vector at once
    for (int VSI = 0; VSI < VS; VSI++) {
      for (int OffsetI = 0; OffsetI < NOffsets; OffsetI++) {
        int Offset = OffsetI * VS;
        Tuint ExpectedI = InI[I + Offset + VSI];
        size_t OutIndex = I + VSI * NOffsets + OffsetI;
        Tuint ComputedI = OutI[OutIndex];

        if (ExpectedI != ComputedI && ++NumErrors < 16) {
          std::cerr << "Error at index=" << OutIndex
                    << ": Expected=" << ExpectedI << ", Computed=" << ComputedI
                    << std::endl;
        }
      }
    }
  }
  return NumErrors == 0;
}

template <typename T, uint16_t N, uint16_t VS, bool UseMask,
          typename PropertiesT>
bool testUSM(queue Q, uint32_t MaskStride, PropertiesT) {

  static_assert(VS > 0 && N % VS == 0,
                "Incorrect VS parameter. N must be divisible by VS.");
  constexpr int NOffsets = N / VS;

  uint32_t Groups = 8;
  uint32_t Threads = 16;

  std::cout << "Running case: T=" << esimd_test::type_name<T>() << ", N=" << N
            << ", VS=" << VS << ", MaskStride=" << MaskStride
            << ", Groups=" << Groups << ", Threads=" << Threads
            << ", use_mask=" << UseMask << std::endl;

  uint16_t Size = Groups * Threads * N;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *Out = sycl::malloc_shared<T>(Size, Q);
  std::memset(Out, 0, Size * sizeof(T));

  T *In = sycl::malloc_shared<T>(Size * 2, Q);
  for (int I = 0; I < Size; I++)
    In[I] = esimd_test::getRandomValue<T>();

  try {
    Q.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
       int GlobalID = NDI.get_global_id(0);
       PropertiesT Props{};

       uint32_t ByteOffset = GlobalID * N * sizeof(T);
       simd<uint32_t, NOffsets> ByteOffsets(ByteOffset, VS * sizeof(T));
       simd_view ByteOffsetsView = ByteOffsets.template select<NOffsets, 1>();

       simd_mask<NOffsets> Pred;
       simd_mask<1> Pred_1 = 1;
       for (int I = 0; I < NOffsets; I++)
         Pred[I] = (I % MaskStride == 0) ? 1 : 0;

       simd<T, N> Vals;
       if constexpr (VS > 1) { // VS > 1 requires specifying <T, N, VS>
         if constexpr (UseMask) {
           if constexpr (sizeof(T) >= 4) {
             if (GlobalID % 4 == 0) // ByteOffset - simd
               prefetch<T, N, VS>(In, ByteOffsets, Pred, Props);
             else if (GlobalID % 4 == 1)
               __ESIMD_NS::prefetch<T, VS>(In, Pred_1, Props);
             else if (GlobalID % 4 == 2)
               __ESIMD_NS::prefetch<T, VS>(In, ByteOffset, Pred_1, Props);
             else // ByteOffset - simd_view
               prefetch<T, N, VS>(In, ByteOffsetsView, Pred, Props);
           } else {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               prefetch<T, N, VS>(In, ByteOffsets, Pred, Props);
             else // ByteOffset - simd_view
               prefetch<T, N, VS>(In, ByteOffsetsView, Pred, Props);
           }
         } else { // UseMask is false
           if constexpr (sizeof(T) >= 4) {
             if (GlobalID % 4 == 0) // ByteOffset - simd
               prefetch<T, N, VS>(In, ByteOffsets, Props);
             else if (GlobalID % 4 == 1)
               __ESIMD_NS::prefetch<T, VS>(In, Props);
             else if (GlobalID % 4 == 2)
               __ESIMD_NS::prefetch<T, VS>(In, ByteOffset, Props);
             else // ByteOffset - simd_view
               prefetch<T, N, VS>(In, ByteOffsetsView, Props);
           } else {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               prefetch<T, N, VS>(In, ByteOffsets, Props);
             else // ByteOffset - simd_view
               prefetch<T, N, VS>(In, ByteOffsetsView, Props);
           }
         }
       } else {
         // if (VS == 1) then <T, N, VS> can often be omitted - test it here.
         // C++ FE do simd to simd_view matching.
         if constexpr (UseMask) {
           if constexpr (sizeof(T) >= 4) {
             if (GlobalID % 4 == 0) // ByteOffset - simd
               prefetch(In, ByteOffsets, Pred, Props);
             else if (GlobalID % 4 == 1)
               __ESIMD_NS::prefetch(In, Pred_1, Props);
             else if (GlobalID % 4 == 2)
               __ESIMD_NS::prefetch(In, ByteOffset, Pred_1, Props);
             else // ByteOffset - simd_view
               prefetch<T, N>(In, ByteOffsetsView, Pred, Props);
           } else {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               prefetch(In, ByteOffsets, Pred, Props);
             else // ByteOffset - simd_view
               prefetch<T, N>(In, ByteOffsetsView, Pred, Props);
           }
         } else { // UseMask is false
           if constexpr (sizeof(T) >= 4) {
             if (GlobalID % 4 == 0) // ByteOffset - simd
               __ESIMD_NS::prefetch(In, ByteOffsets, Props);
             else if (GlobalID % 4 == 1)
               __ESIMD_NS::prefetch(In, Props);
             else if (GlobalID % 4 == 2)
               __ESIMD_NS::prefetch(In, ByteOffset, Props);
             else // ByteOffset - simd_view
               prefetch<T, N>(In, ByteOffsetsView, Props);
           } else {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               __ESIMD_NS::prefetch(In, ByteOffsets, Props);
             else // ByteOffset - simd_view
               prefetch<T, N>(In, ByteOffsetsView, Props);
           }
         }
       } // end if (VS == 1)
       Vals = gather<T, N, VS>(In, ByteOffsets);
       Vals.copy_to(Out + GlobalID * N);
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(In, Q);
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(In, Out, N, Size, VS);
  if (!Passed)
    std::cout << "Case FAILED" << std::endl;

  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T> bool testUSM(queue Q) {
  constexpr bool UseMask = true;

  properties CacheProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>};

  bool Passed = true;
  Passed &= testUSM<T, 1, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 2, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 4, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 8, 1, !UseMask>(Q, 3, CacheProps);
  Passed &= testUSM<T, 16, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 32, 1, !UseMask>(Q, 2, CacheProps);

  Passed &= testUSM<T, 1, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 2, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 4, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 8, 1, UseMask>(Q, 3, CacheProps);
  Passed &= testUSM<T, 16, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testUSM<T, 32, 1, UseMask>(Q, 2, CacheProps);

  // Check VS > 1. GPU supports only dwords and qwords in this mode.
  if constexpr (sizeof(T) >= 4) {
    Passed &= testUSM<T, 16, 2, UseMask>(Q, 3, CacheProps);
    Passed &= testUSM<T, 16, 2, !UseMask>(Q, 3, CacheProps);
    Passed &= testUSM<T, 32, 2, !UseMask>(Q, 3, CacheProps);
    Passed &= testUSM<T, 32, 2, UseMask>(Q, 3, CacheProps);
  }
  return Passed;
}

template <typename T, uint16_t N, uint16_t VS, bool UseMask,
          typename PropertiesT>
bool testACC(queue Q, uint32_t MaskStride, PropertiesT) {

  static_assert(VS > 0 && N % VS == 0,
                "Incorrect VS parameter. N must be divisible by VS.");
  constexpr int NOffsets = N / VS;

  uint32_t Groups = 8;
  uint32_t Threads = 16;

  std::cout << "Running case: T=" << esimd_test::type_name<T>() << ", N=" << N
            << ", VS=" << VS << ", MaskStride=" << MaskStride
            << ", Groups=" << Groups << ", Threads=" << Threads
            << ", use_mask=" << UseMask << std::endl;

  uint16_t Size = Groups * Threads * N;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *Out = sycl::malloc_shared<T>(Size, Q);
  std::memset(Out, 0, Size * sizeof(T));

  T *In = sycl::malloc_shared<T>(Size * 2, Q);
  for (int I = 0; I < Size; I++)
    In[I] = esimd_test::getRandomValue<T>();

  try {
    buffer<T, 1> InBuf(In, Size * 2);
    Q.submit([&](handler &CGH) {
       accessor InAcc{InBuf, CGH};
       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         int GlobalID = NDI.get_global_id(0);
         PropertiesT Props{};

         uint32_t ByteOffset = GlobalID * N * sizeof(T);
         simd<uint32_t, NOffsets> ByteOffsets(ByteOffset, VS * sizeof(T));
         simd_view ByteOffsetsView = ByteOffsets.template select<NOffsets, 1>();

         simd_mask<NOffsets> Pred;
         simd_mask<1> Pred_1 = 1;
         for (int I = 0; I < NOffsets; I++)
           Pred[I] = (I % MaskStride == 0) ? 1 : 0;

         simd<T, N> Vals;
         if constexpr (VS > 1) { // VS > 1 requires specifying <T, N, VS>
           if constexpr (UseMask) {
             if constexpr (sizeof(T) >= 4) {
               if (GlobalID % 4 == 0) // ByteOffset - simd
                 prefetch<T, N, VS>(InAcc, ByteOffsets, Pred, Props);
               else if (GlobalID % 4 == 1)
                 prefetch<T, VS>(InAcc, Pred_1, Props);
               else if (GlobalID % 4 == 2)
                 prefetch<T, VS>(InAcc, ByteOffset, Pred_1, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N, VS>(InAcc, ByteOffsetsView, Pred, Props);
             } else {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 prefetch<T, N, VS>(InAcc, ByteOffsets, Pred, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N, VS>(InAcc, ByteOffsetsView, Pred, Props);
             }
           } else { // UseMask is false
             if constexpr (sizeof(T) >= 4) {
               if (GlobalID % 4 == 0) // ByteOffset - simd
                 prefetch<T, N, VS>(InAcc, ByteOffsets, Props);
               else if (GlobalID % 4 == 1)
                 prefetch<T, VS>(InAcc, Props);
               else if (GlobalID % 4 == 2)
                 prefetch<T, VS>(InAcc, ByteOffset, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N, VS>(InAcc, ByteOffsetsView, Props);
             } else {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 prefetch<T, N, VS>(InAcc, ByteOffsets, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N, VS>(InAcc, ByteOffsetsView, Props);
             }
           }
         } else {
           // if (VS == 1) then <T, N, VS> can often be omitted - test it
           // here. C++ FE do simd to simd_view matching.
           if constexpr (UseMask) {
             if constexpr (sizeof(T) >= 4) {
               if (GlobalID % 4 == 0) // ByteOffset - simd
                 prefetch<T>(InAcc, ByteOffsets, Pred, Props);
               else if (GlobalID % 4 == 1)
                 prefetch<T>(InAcc, Pred_1, Props);
               else if (GlobalID % 4 == 2)
                 prefetch<T>(InAcc, ByteOffset, Pred_1, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N>(InAcc, ByteOffsetsView, Pred, Props);
             } else {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 prefetch<T>(InAcc, ByteOffsets, Pred, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N>(InAcc, ByteOffsetsView, Pred, Props);
             }
           } else { // UseMask is false
             if constexpr (sizeof(T) >= 4) {
               if (GlobalID % 4 == 0) // ByteOffset - simd
                 prefetch<T>(InAcc, ByteOffsets, Props);
               else if (GlobalID % 4 == 1)
                 prefetch<T>(InAcc, Props);
               else if (GlobalID % 4 == 2)
                 prefetch<T>(InAcc, ByteOffset, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N>(InAcc, ByteOffsetsView, Props);
             } else {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 prefetch<T>(InAcc, ByteOffsets, Props);
               else // ByteOffset - simd_view
                 prefetch<T, N>(InAcc, ByteOffsetsView, Props);
             }
           }
         } // end if (VS == 1)
         Vals = gather<T, N, VS>(InAcc, ByteOffsets);
         Vals.copy_to(Out + GlobalID * N);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(In, Q);
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(In, Out, N, Size, VS);
  if (!Passed)
    std::cout << "Case FAILED" << std::endl;

  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T> bool testACC(queue Q) {
  constexpr bool UseMask = true;

  properties CacheProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>};

  bool Passed = true;
  Passed &= testACC<T, 1, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 2, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 4, 1, !UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 8, 1, !UseMask>(Q, 3, CacheProps);
  Passed &= testACC<T, 16, 1, !UseMask>(Q, 3, CacheProps);
  Passed &= testACC<T, 32, 1, !UseMask>(Q, 3, CacheProps);
  Passed &= testACC<T, 64, 1, !UseMask>(Q, 3, CacheProps);

  Passed &= testACC<T, 1, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 2, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 4, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 8, 1, UseMask>(Q, 3, CacheProps);
  Passed &= testACC<T, 16, 1, UseMask>(Q, 3, CacheProps);
  Passed &= testACC<T, 32, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 64, 1, UseMask>(Q, 2, CacheProps);
  Passed &= testACC<T, 6, 1, UseMask>(Q, 3, CacheProps);

  // Check VS > 1. GPU supports only dwords and qwords in this mode.
  if constexpr (sizeof(T) >= 4) {
    Passed &= testACC<T, 16, 2, UseMask>(Q, 3, CacheProps);
    Passed &= testACC<T, 32, 2, !UseMask>(Q, 3, CacheProps);
    Passed &= testACC<T, 64, 2, !UseMask>(Q, 3, CacheProps);
    Passed &= testACC<T, 64, 2, UseMask>(Q, 3, CacheProps);
    Passed &= testACC<T, 6, 2, UseMask>(Q, 3, CacheProps);
  }
  return Passed;
}
