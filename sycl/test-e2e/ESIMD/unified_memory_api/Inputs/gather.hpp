//==------------- gather.hpp - DPC++ ESIMD on-device test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t OffsetT;
#else
typedef uint32_t OffsetT;
#endif

template <typename T>
bool verify(const T *In, const T *Out, int N, int Size, int VS,
            uint32_t MaskStride, bool UseMask, bool UsePassThru) {
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

        bool IsMaskSet = UseMask ? ((OffsetI % MaskStride) == 0 ? 1 : 0) : true;
        if (!IsMaskSet) {
          if (!UsePassThru)
            continue; // Value is undefined;
          ExpectedI = OutIndex;
        }

        if (ExpectedI != ComputedI && ++NumErrors < 16) {
          std::cerr << "Error at index=" << OutIndex
                    << ": Expected=" << ExpectedI << ", Computed=" << ComputedI
                    << ", IsMaskSet=" << IsMaskSet << std::endl;
        }
      }
    }
  }
  return NumErrors == 0;
}

template <typename T, uint16_t N, uint16_t VS, bool UseMask, bool UsePassThru,
          bool UseProperties, typename PropertiesT>
bool testUSM(queue Q, uint32_t MaskStride, PropertiesT) {

  static_assert(VS > 0 && N % VS == 0,
                "Incorrect VS parameter. N must be divisible by VS.");
  constexpr int NOffsets = N / VS;
  static_assert(!UsePassThru || UseMask,
                "PassThru cannot be used without using mask");

  uint32_t Groups = 8;
  uint32_t Threads = 16;

  std::cout << "Running case: T=" << esimd_test::type_name<T>() << ", N=" << N
            << ", VS=" << VS << ", MaskStride=" << MaskStride
            << ", Groups=" << Groups << ", Threads=" << Threads
            << ", use_mask=" << UseMask << ", use_pass_thru=" << UsePassThru
            << ", use_properties=" << UseProperties << std::endl;

  uint16_t Size = Groups * Threads * N;
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;

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

       simd<OffsetT, NOffsets> ByteOffsets(GlobalID * N * sizeof(T),
                                           VS * sizeof(T));
       simd_view ByteOffsetsView = ByteOffsets.template select<NOffsets, 1>();

       simd_mask<NOffsets> Pred;
       for (int I = 0; I < NOffsets; I++)
         Pred[I] = (I % MaskStride == 0) ? 1 : 0;

       using Tuint = esimd_test::uint_type_t<sizeof(T)>;
       simd<Tuint, N> PassThruInt(GlobalID * N, 1);
       simd<T, N> PassThru = PassThruInt.template bit_cast_view<T>();
       auto PassThruView = PassThru.template select<N, 1>(0);

       simd<T, N> Vals;
       if constexpr (VS > 1) { // VS > 1 requires specifying <T, N, VS>
         if constexpr (UsePassThru) {
           if constexpr (UseProperties) {
             if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
               Vals = gather<T, N, VS>(In, ByteOffsets, Pred, PassThru, Props);
             else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
               Vals =
                   gather<T, N, VS>(In, ByteOffsets, Pred, PassThruView, Props);
             else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
               Vals =
                   gather<T, N, VS>(In, ByteOffsetsView, Pred, PassThru, Props);
             else // ByteOffset - view, PassThru - view
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Pred, PassThruView,
                                       Props);
           } else {                 // UseProperties is false
             if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
               Vals = gather<T, N, VS>(In, ByteOffsets, Pred, PassThru);
             else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
               Vals = gather<T, N, VS>(In, ByteOffsets, Pred, PassThruView);
             else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Pred, PassThru);
             else // ByteOffset - view, PassThru - view
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Pred, PassThruView);
           }
         } else if constexpr (UseMask) { // UsePassThru is false
           if constexpr (UseProperties) {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather<T, N, VS>(In, ByteOffsets, Pred, Props);
             else // ByteOffset - simd_view
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Pred, Props);
           } else {                 // UseProperties is false
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather<T, N, VS>(In, ByteOffsets, Pred);
             else // ByteOffset - simd_view
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Pred);
           }
         } else { // UseMask is false, UsePassThru is false
           if constexpr (UseProperties) {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather<T, N, VS>(In, ByteOffsets, Props);
             else // ByteOffset - simd_view
               Vals = gather<T, N, VS>(In, ByteOffsetsView, Props);
           } else {                 // UseProperties is false
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather<T, N, VS>(In, ByteOffsets);
             else // ByteOffset - simd_view
               Vals = gather<T, N, VS>(In, ByteOffsetsView);
           }
         }
       } else {
         // if (VS == 1) then <T, N, VS> can often be omitted - test it here.
         // The variants accepting simd_view for 'PassThru' operand though
         // still require <T, N> to be specified explicitly to help
         // C++ FE do simd to simd_view matching.
         if constexpr (UsePassThru) {
           if constexpr (UseProperties) {
             if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
               Vals = gather(In, ByteOffsets, Pred, PassThru, Props);
             else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
               Vals = gather<T, N>(In, ByteOffsets, Pred, PassThruView, Props);
             else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
               Vals = gather(In, ByteOffsetsView, Pred, PassThru, Props);
             else // ByteOffset - view, PassThru - view
               Vals =
                   gather<T, N>(In, ByteOffsetsView, Pred, PassThruView, Props);
           } else {                 // UseProperties is false
             if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
               Vals = gather(In, ByteOffsets, Pred, PassThru);
             else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
               Vals = gather<T, N>(In, ByteOffsets, Pred, PassThruView);
             else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
               Vals = gather<T, N>(In, ByteOffsetsView, Pred, PassThru);
             else // ByteOffset - view, PassThru - view
               Vals = gather<T, N>(In, ByteOffsetsView, Pred, PassThruView);
           }
         } else if constexpr (UseMask) { // UsePassThru is false
           if constexpr (UseProperties) {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather(In, ByteOffsets, Pred, Props);
             else // ByteOffset - simd_view
               Vals = gather<T, N>(In, ByteOffsetsView, Pred, Props);
           } else {                 // UseProperties is false
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather(In, ByteOffsets, Pred);
             else // ByteOffset - simd_view
               Vals = gather<T, N>(In, ByteOffsetsView, Pred);
           }
         } else { // UsePassThru is false, UseMask is false
           if constexpr (UseProperties) {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather(In, ByteOffsets, Props);
             else // ByteOffset - simd_view
               Vals = gather<T, N>(In, ByteOffsetsView, Props);
           } else {
             if (GlobalID % 2 == 0) // ByteOffset - simd
               Vals = gather(In, ByteOffsets);
             else // ByteOffset - simd_view
               Vals = gather<T, N>(In, ByteOffsetsView);
           }
         }
       } // end if (VS == 1)
       Vals.copy_to(Out + GlobalID * N);
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(In, Q);
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(In, Out, N, Size, VS, MaskStride, UseMask, UsePassThru);
  if (!Passed)
    std::cout << "Case FAILED" << std::endl;

  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T, uint16_t N, uint16_t VS, bool UseMask, bool UsePassThru,
          bool UseProperties, typename PropertiesT>
bool testACC(queue Q, uint32_t MaskStride, PropertiesT) {

  static_assert(VS > 0 && N % VS == 0,
                "Incorrect VS parameter. N must be divisible by VS.");
  constexpr int NOffsets = N / VS;
  static_assert(!UsePassThru || UseMask,
                "PassThru cannot be used without using mask");

  uint32_t Groups = 8;
  uint32_t Threads = 16;

  std::cout << "Running case: T=" << esimd_test::type_name<T>() << ", N=" << N
            << ", VS=" << VS << ", MaskStride=" << MaskStride
            << ", Groups=" << Groups << ", Threads=" << Threads
            << ", use_mask=" << UseMask << ", use_pass_thru=" << UsePassThru
            << ", use_properties=" << UseProperties << std::endl;

  uint16_t Size = Groups * Threads * N;
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;

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

         simd<OffsetT, NOffsets> ByteOffsets(GlobalID * N * sizeof(T),
                                             VS * sizeof(T));
         simd_view ByteOffsetsView = ByteOffsets.template select<NOffsets, 1>();

         simd_mask<NOffsets> Pred;
         for (int I = 0; I < NOffsets; I++)
           Pred[I] = (I % MaskStride == 0) ? 1 : 0;

         using Tuint = esimd_test::uint_type_t<sizeof(T)>;
         simd<Tuint, N> PassThruInt(GlobalID * N, 1);
         simd<T, N> PassThru = PassThruInt.template bit_cast_view<T>();
         auto PassThruView = PassThru.template select<N, 1>(0);

         simd<T, N> Vals;
         if constexpr (VS > 1) { // VS > 1 requires specifying <T, N, VS>
           if constexpr (UsePassThru) {
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Pred, PassThru,
                                         Props);
               else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Pred, PassThruView,
                                         Props);
               else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Pred, PassThru,
                                         Props);
               else // ByteOffset - view, PassThru - view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Pred,
                                         PassThruView, Props);
             } else {                 // UseProperties is false
               if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Pred, PassThru);
               else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
                 Vals =
                     gather<T, N, VS>(InAcc, ByteOffsets, Pred, PassThruView);
               else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
                 Vals =
                     gather<T, N, VS>(InAcc, ByteOffsetsView, Pred, PassThru);
               else // ByteOffset - view, PassThru - view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Pred,
                                         PassThruView);
             }
           } else if constexpr (UseMask) { // UsePassThru is false
             if constexpr (UseProperties) {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Pred, Props);
               else // ByteOffset - simd_view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Pred, Props);
             } else {                 // UseProperties is false
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Pred);
               else // ByteOffset - simd_view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Pred);
             }
           } else { // UseMask is false, UsePassThru is false
             if constexpr (UseProperties) {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets, Props);
               else // ByteOffset - simd_view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView, Props);
             } else {                 // UseProperties is false
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T, N, VS>(InAcc, ByteOffsets);
               else // ByteOffset - simd_view
                 Vals = gather<T, N, VS>(InAcc, ByteOffsetsView);
             }
           }
         } else {
           // if (VS == 1) then <T, N, VS> can often be omitted - test it here.
           // The variants accepting simd_view for 'PassThru' operand though
           // still require <T, N> to be specified explicitly to help
           // C++ FE do simd to simd_view matching.
           if constexpr (UsePassThru) {
             if constexpr (UseProperties) {
               if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
                 Vals = gather<T>(InAcc, ByteOffsets, Pred, PassThru, Props);
               else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
                 Vals = gather<T, N>(InAcc, ByteOffsets, Pred, PassThruView,
                                     Props);
               else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
                 Vals = gather(InAcc, ByteOffsetsView, Pred, PassThru, Props);
               else // ByteOffset - view, PassThru - view
                 Vals = gather<T, N>(InAcc, ByteOffsetsView, Pred, PassThruView,
                                     Props);
             } else {                 // UseProperties is false
               if (GlobalID % 4 == 0) // ByteOffset - simd, PassThru - simd
                 Vals = gather(InAcc, ByteOffsets, Pred, PassThru);
               else if (GlobalID % 4 == 1) // ByteOffset - simd, PassThru - view
                 Vals = gather<T, N>(InAcc, ByteOffsets, Pred, PassThruView);
               else if (GlobalID % 4 == 2) // ByteOffset - view, PassThru - simd
                 Vals = gather<T, N>(InAcc, ByteOffsetsView, Pred, PassThru);
               else // ByteOffset - view, PassThru - view
                 Vals =
                     gather<T, N>(InAcc, ByteOffsetsView, Pred, PassThruView);
             }
           } else if constexpr (UseMask) { // UsePassThru is false
             if constexpr (UseProperties) {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T>(InAcc, ByteOffsets, Pred, Props);
               else // ByteOffset - simd_view
                 Vals = gather<T, N>(InAcc, ByteOffsetsView, Pred, Props);
             } else {                 // UseProperties is false
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T>(InAcc, ByteOffsets, Pred);
               else // ByteOffset - simd_view
                 Vals = gather<T, N>(InAcc, ByteOffsetsView, Pred);
             }
           } else { // UsePassThru is false, UseMask is false
             if constexpr (UseProperties) {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T>(InAcc, ByteOffsets, Props);
               else // ByteOffset - simd_view
                 Vals = gather<T, N>(InAcc, ByteOffsetsView, Props);
             } else {
               if (GlobalID % 2 == 0) // ByteOffset - simd
                 Vals = gather<T>(InAcc, ByteOffsets);
               else // ByteOffset - simd_view
                 Vals = gather<T, N>(InAcc, ByteOffsetsView);
             }
           }
         } // end if (VS == 1)
         Vals.copy_to(Out + GlobalID * N);
         // scatter(Out, ByteOffsets.template select<NOffsets, 1>(), Vals);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(In, Q);
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(In, Out, N, Size, VS, MaskStride, UseMask, UsePassThru);
  if (!Passed)
    std::cout << "Case FAILED" << std::endl;

  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T, TestFeatures Features> bool testUSM(queue Q) {
  constexpr bool UseMask = true;
  constexpr bool UsePassThru = true;
  constexpr bool UseProperties = true;

  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;
  Passed &= testUSM<T, 1, 1, !UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testUSM<T, 2, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testUSM<T, 4, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testUSM<T, 8, 1, UseMask, !UsePassThru, UseProperties>(
      Q, 3, AlignElemProps);
#ifdef __ESIMD_GATHER_SCATTER_LLVM_IR
  Passed &= testUSM<T, 16, 1, UseMask, UsePassThru, UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testUSM<T, 32, 1, UseMask, UsePassThru, !UseProperties>(
      Q, 3, AlignElemProps);
#endif

  // TODO: test non-power-of-2 N
  // Such cases were promised to be supported, but in fact they fail.
  // Create some test cases here after the issue in GPU driver is resolved.

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    properties LSCProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>,
                        alignment<sizeof(T)>};
    Passed &=
        testUSM<T, 1, 1, !UseMask, !UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testUSM<T, 2, 1, UseMask, !UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testUSM<T, 4, 1, UseMask, UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testUSM<T, 8, 1, UseMask, UsePassThru, UseProperties>(Q, 3, LSCProps);

    Passed &=
        testUSM<T, 32, 1, UseMask, UsePassThru, UseProperties>(Q, 2, LSCProps);

    // Check VS > 1. GPU supports only dwords and qwords in this mode.
    if constexpr (sizeof(T) >= 4) {
      // TODO: This test case causes flaky fail. Enable it after the issue
      // in GPU driver is fixed.
      // Passed &= testUSM<T, 16, 2, UseMask, !UsePassThru, UseProperties>(
      //    Q, 3, AlignElemProps);

      Passed &= testUSM<T, 32, 2, !UseMask, !UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
      Passed &= testUSM<T, 32, 2, UseMask, !UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
      Passed &= testUSM<T, 32, 2, UseMask, UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
    }
  }
  return Passed;
}

template <typename T, TestFeatures Features> bool testACC(queue Q) {
  constexpr bool UseMask = true;
  constexpr bool UsePassThru = true;
  constexpr bool UseProperties = true;

  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;
  Passed &= testACC<T, 1, 1, !UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
#ifdef __ESIMD_FORCE_STATELESS_MEM
  Passed &= testACC<T, 2, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testACC<T, 4, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 2, AlignElemProps);
#endif // __ESIMD_FORCE_STATELESS_MEM
  Passed &= testACC<T, 8, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 3, AlignElemProps);
  Passed &= testACC<T, 16, 1, UseMask, !UsePassThru, UseProperties>(
      Q, 2, AlignElemProps);
  Passed &= testACC<T, 32, 1, UseMask, !UsePassThru, !UseProperties>(
      Q, 3, AlignElemProps);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    properties LSCProps{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::cached>,
                        alignment<sizeof(T)>};
    Passed &=
        testACC<T, 1, 1, !UseMask, !UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testACC<T, 2, 1, UseMask, !UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testACC<T, 4, 1, UseMask, UsePassThru, UseProperties>(Q, 2, LSCProps);
    Passed &=
        testACC<T, 8, 1, UseMask, UsePassThru, UseProperties>(Q, 3, LSCProps);

    Passed &=
        testACC<T, 32, 1, UseMask, UsePassThru, UseProperties>(Q, 2, LSCProps);

    // Check VS > 1. GPU supports only dwords and qwords in this mode.
    if constexpr (sizeof(T) >= 4) {
      // TODO: This test case causes flaky fail. Enable it after the issue
      // in GPU driver is fixed.
      // Passed &= testACC<T, 16, 2, UseMask, !UsePassThru, UseProperties>(
      //    Q, 3, AlignElemProps);

      Passed &= testACC<T, 32, 2, !UseMask, !UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
      Passed &= testACC<T, 32, 2, UseMask, !UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
      Passed &= testACC<T, 32, 2, UseMask, UsePassThru, UseProperties>(
          Q, 3, AlignElemProps);
    }
  }
  return Passed;
}
