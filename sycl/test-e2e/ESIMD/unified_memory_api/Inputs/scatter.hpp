//==------- scatter.hpp - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T>
bool verify(const T *Out, int N, int Size, int VS, uint32_t MaskStride,
            bool UseMask) {
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;
  int NumErrors = 0;
  int NOffsets = N / VS;
  for (uint32_t I = 0; I < Size; I += N) { // Verify by 1 vector at once
    for (int VSI = 0; VSI < VS; VSI++) {
      for (int OffsetI = 0; OffsetI < NOffsets; OffsetI++) {
        size_t OutIndex = I + VSI * NOffsets + OffsetI;
        bool IsMaskSet = UseMask ? ((OutIndex / VS) % MaskStride == 0) : true;
        Tuint Expected = sycl::bit_cast<Tuint>((T)OutIndex);
        if (!UseMask || IsMaskSet)
          Expected = sycl::bit_cast<Tuint>((T)(OutIndex * 2));
        Tuint Computed = sycl::bit_cast<Tuint>(Out[OutIndex]);
        if (Computed != Expected && ++NumErrors < 16) {
          std::cout << "Out[" << OutIndex << "] = " << std::to_string(Computed)
                    << " vs " << std::to_string(Expected) << std::endl;
        }
      }
    }
  }
  return NumErrors == 0;
}

template <typename T, uint16_t N, uint16_t VS, bool UseMask, bool UseProperties,
          typename ScatterPropertiesT>
bool testUSM(queue Q, uint32_t MaskStride,
             ScatterPropertiesT ScatterProperties) {
  uint32_t Groups = 8;
  uint32_t Threads = 16;
  size_t Size = Groups * Threads * N;
  static_assert(VS > 0 && N % VS == 0,
                "Incorrect VS parameter. N must be divisible by VS.");
  constexpr int NOffsets = N / VS;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;

  std::cout << "USM case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ", VS=" << VS << ",UseMask=" << UseMask
            << ",UseProperties=" << UseProperties << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *Out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), Q));
  for (size_t i = 0; i < Size; i++)
    Out[i] = i;

  try {
    Q.submit([&](handler &cgh) {
       cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         ScatterPropertiesT Props{};
         uint16_t GlobalID = ndi.get_global_id(0);
         simd<int32_t, NOffsets> ByteOffsets(GlobalID * N * sizeof(T),
                                             VS * sizeof(T));
         auto ByteOffsetsView = ByteOffsets.template select<NOffsets, 1>();
         simd<T, N> Vals = gather<T, N, VS>(Out, ByteOffsets);
         Vals *= 2;
         auto ValsView = Vals.template select<N, 1>();
         simd_mask<NOffsets> Pred = 0;
         for (int I = 0; I < NOffsets; I++)
           Pred[I] = (I % MaskStride == 0) ? 1 : 0;
         if constexpr (VS > 1) { // VS > 1 requires specifying <T, N, VS>
           if constexpr (UseMask) {
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0)
                 scatter<T, N, VS>(Out, ByteOffsets, Vals, Pred, Props);
               else if (GlobalID % 4 == 1)
                 scatter<T, N, VS>(Out, ByteOffsetsView, Vals, Pred, Props);
               else if (GlobalID % 4 == 2)
                 scatter<T, N, VS>(Out, ByteOffsets, ValsView, Pred, Props);
               else if (GlobalID % 4 == 3)
                 scatter<T, N, VS>(Out, ByteOffsetsView, ValsView, Pred, Props);
             } else { // UseProperties == false
               if (GlobalID % 4 == 0)
                 scatter<T, N, VS>(Out, ByteOffsets, Vals, Pred);
               else if (GlobalID % 4 == 1)
                 scatter<T, N, VS>(Out, ByteOffsetsView, Vals, Pred);
               else if (GlobalID % 4 == 2)
                 scatter<T, N, VS>(Out, ByteOffsets, ValsView, Pred);
               else if (GlobalID % 4 == 3)
                 scatter<T, N, VS>(Out, ByteOffsetsView, ValsView, Pred);
             }
           } else { // UseMask == false
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0)
                 scatter<T, N, VS>(Out, ByteOffsets, Vals, Props);
               else if (GlobalID % 4 == 1)
                 scatter<T, N, VS>(Out, ByteOffsetsView, Vals, Props);
               else if (GlobalID % 4 == 2)
                 scatter<T, N, VS>(Out, ByteOffsets, ValsView, Props);
               else if (GlobalID % 4 == 3)
                 scatter<T, N, VS>(Out, ByteOffsetsView, ValsView, Props);
             } else { // UseProperties == false
               if (GlobalID % 4 == 0)
                 scatter<T, N, VS>(Out, ByteOffsets, Vals);
               else if (GlobalID % 4 == 1)
                 scatter<T, N, VS>(Out, ByteOffsetsView, Vals);
               else if (GlobalID % 4 == 2)
                 scatter<T, N, VS>(Out, ByteOffsets, ValsView);
               else if (GlobalID % 4 == 3)
                 scatter<T, N, VS>(Out, ByteOffsetsView, ValsView);
             }
           }
         } else { // VS == 1
           if constexpr (UseMask) {
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0)
                 scatter(Out, ByteOffsets, Vals, Pred, Props);
               else if (GlobalID % 4 == 1)
                 scatter(Out, ByteOffsetsView, Vals, Pred, Props);
               else if (GlobalID % 4 == 2)
                 scatter<T, N>(Out, ByteOffsets, ValsView, Pred, Props);
               else if (GlobalID % 4 == 3)
                 scatter<T, N>(Out, ByteOffsetsView, ValsView, Pred, Props);
             } else { // UseProperties == false
               if (GlobalID % 4 == 0)
                 scatter(Out, ByteOffsets, Vals, Pred);
               else if (GlobalID % 4 == 1)
                 scatter(Out, ByteOffsetsView, Vals, Pred);
               else if (GlobalID % 4 == 2)
                 scatter(Out, ByteOffsets, ValsView, Pred);
               else if (GlobalID % 4 == 3)
                 scatter(Out, ByteOffsetsView, ValsView, Pred);
             }
           } else { // UseMask == false
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0)
                 scatter(Out, ByteOffsets, Vals, Props);
               else if (GlobalID % 4 == 1)
                 scatter(Out, ByteOffsetsView, Vals, Props);
               else if (GlobalID % 4 == 2)
                 scatter<T, N>(Out, ByteOffsets, ValsView, Props);
               else if (GlobalID % 4 == 3)
                 scatter<T, N>(Out, ByteOffsetsView, ValsView, Props);
             } else { // UseProperties == false
               if (GlobalID % 4 == 0)
                 scatter(Out, ByteOffsets, Vals);
               else if (GlobalID % 4 == 1)
                 scatter(Out, ByteOffsetsView, Vals);
               else if (GlobalID % 4 == 2)
                 scatter<T, N>(Out, ByteOffsets, ValsView);
               else if (GlobalID % 4 == 3)
                 scatter<T, N>(Out, ByteOffsetsView, ValsView);
             }
           }
         }
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(Out, N, Size, VS, MaskStride, UseMask);

  sycl::free(Out, Q);

  return Passed;
}

template <typename T, TestFeatures Features> bool testUSM(queue Q) {
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;
  properties EmptyProps;
  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;

  // // Test scatter() that is available on Gen12 and PVC.
  Passed &= testUSM<T, 1, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);
  Passed &= testUSM<T, 2, 1, !CheckMask, CheckProperties>(Q, 1, EmptyProps);
  Passed &= testUSM<T, 1, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);
  Passed &= testUSM<T, 8, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);
  Passed &= testUSM<T, 16, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);
  Passed &= testUSM<T, 16, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);

  Passed &= testUSM<T, 32, 1, !CheckMask, CheckProperties>(Q, 2, EmptyProps);

  // // Test scatter() without passing compile-time properties argument.
  Passed &= testUSM<T, 16, 1, !CheckMask, !CheckProperties>(Q, 2, EmptyProps);
  Passed &= testUSM<T, 32, 1, !CheckMask, !CheckProperties>(Q, 2, EmptyProps);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    properties LSCProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::uncached>,
                        alignment<sizeof(T)>};
    Passed &= testUSM<T, 1, 1, !CheckMask, CheckProperties>(Q, 2, LSCProps);
    Passed &= testUSM<T, 2, 1, CheckMask, CheckProperties>(Q, 2, LSCProps);
    Passed &= testUSM<T, 4, 1, CheckMask, CheckProperties>(Q, 2, LSCProps);
    Passed &= testUSM<T, 8, 1, CheckMask, CheckProperties>(Q, 2, LSCProps);

    Passed &= testUSM<T, 32, 1, CheckMask, CheckProperties>(Q, 2, LSCProps);

    // Check VS > 1. GPU supports only dwords and qwords in this mode.
    if constexpr (sizeof(T) >= 4) {
      // TODO: This test case causes flaky fail. Enable it after the issue
      // in GPU driver is fixed.
      // Passed &=
      //     testUSM<T, 16, 2, CheckMask, CheckProperties>(Q, 2, AlignElemProps)
      Passed &=
          testUSM<T, 32, 2, !CheckMask, CheckProperties>(Q, 2, AlignElemProps);
      Passed &=
          testUSM<T, 32, 2, CheckMask, CheckProperties>(Q, 2, AlignElemProps);
      Passed &=
          testUSM<T, 32, 2, CheckMask, CheckProperties>(Q, 2, AlignElemProps);
    }
  } // TestPVCFeatures

  return Passed;
}
