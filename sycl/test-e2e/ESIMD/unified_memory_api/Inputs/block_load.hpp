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
bool verify(const T *In, const T *Out, size_t Size, int N, bool UseMask,
            bool UsePassThruOperand) {
  int NumErrors = 0;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;
  for (int i = 0; i < Size && NumErrors < 32; i++) {
    bool IsMaskSet = UseMask ? ((i / N + 1) & 0x1) : true;
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
                << " vs etalon = 0x" << Expected << std::dec
                << ", IsMaskSet = " << IsMaskSet << std::endl;
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
         simd_mask<1> Mask = (GlobalID + 1) & 0x1;
         // TODO: these 2 lines work-around the problem with scalar
         // conversions to bfloat16. It could be just: "simd<T, N>
         // PassThru(ElemOffset, 1);"
         simd<uint32_t, N> PassThruInt(ElemOffset, 1);
         simd<T, N> PassThru = PassThruInt;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
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

  bool Passed = verify<T>(In, Out, Size, N, UseMask, UsePassThruOperand);
  sycl::free(In, Q);
  sycl::free(Out, Q);
  return Passed;
}

template <typename T, TestFeatures Features> bool testUSM(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;

  // Test block_load() that is available on Gen12, DG2 and PVC.
  Passed &= testUSM<T, 1, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 2, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 1, 4, AlignElemProps);
  Passed &= testUSM<T, 3, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 8, AlignElemProps);
  Passed &= testUSM<T, 4, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 8, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 16, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);

  // Intentionally check non-power-of-2 simd size - it must work.
  // Just pass element-size alignment.
  // These test cases compute wrong values for for the few last elements
  // if the driver is not new enough.
  // TODO: windows version with the fix is not known. Enable it eventually.
  if (sizeof(T) > 2 ||
      esimd_test::isGPUDriverGE(Q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                "27556", "win.just.skip.test", false)) {
    Passed &= testUSM<T, 33, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, 4, AlignElemProps);
    Passed &= testUSM<T, 67, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, 4, AlignElemProps);
  }

  // Intentionally check big simd size - it must work.
  Passed &= testUSM<T, 512, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 1024, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);

  // Test block_load() without passing compile-time properties argument.
  Passed &= testUSM<T, 16, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, 4, Align16Props);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    // Using mask or cache hints adds the requirement to run tests on DG2/PVC.
    // Also, DG2/DG2/PVC variant currently requires a) power-or-two elements,
    // b) the number of bytes loaded per call must not exceed 512,
    // c) the alignment of USM ptr + offset to be 4 or 8-bytes(for 8-byte
    // element vectors).

    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{cache_hint_L1<cache_hint::streaming>,
                             cache_hint_L2<cache_hint::cached>,
                             alignment<RequiredAlignment>};

    // Only d/q-words are supported now.
    // Thus we use this I32Factor for testing purposes and convenience.
    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    Passed &=
        testUSM<T, 1 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    Passed &=
        testUSM<T, 2 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 5, 5, DG2OrPVCProps);
    Passed &=
        testUSM<T, 4 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 5, 5, DG2OrPVCProps);
    Passed &=
        testUSM<T, 8 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 5, 5, DG2OrPVCProps);
    Passed &=
        testUSM<T, 16 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 5, 5, DG2OrPVCProps);
    Passed &=
        testUSM<T, 32 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);

    // This call (potentially) and the next call (guaranteed) load the biggest
    // load-able chunk, which requires loading with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::streaming>,
                              cache_hint_L2<cache_hint::cached>, alignment<8>};
    if constexpr (Features == TestFeatures::PVC) {
      Passed &=
          testUSM<T, 64 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
              Q, 7, 1, PVCAlign8Props);
      if constexpr (sizeof(T) <= 4)
        Passed &=
            testUSM<T, 128 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
                Q, 1, 4, PVCAlign8Props);
    }
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
    buffer<T, 1> InBuf(In);
    Q.submit([&](handler &CGH) {
       accessor InAcc{InBuf, CGH};
       auto OutPtr = Out.data();

       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = NDI.get_global_id(0);
         uint32_t ElemOffset = GlobalID * N;

         simd<T, N> Vals;
         simd_mask<1> Mask = (GlobalID + 1) & 0x1;
         // TODO: these 2 lines work-around the problem with scalar
         // conversions to bfloat16. It could be just: "simd<T, N>
         // PassThru(ElemOffset, 1);"
         simd<uint32_t, N> PassThruInt(ElemOffset, 1);
         simd<T, N> PassThru = PassThruInt;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
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

  bool Passed =
      verify<T>(In.data(), Out.data(), Size, N, UseMask, UsePassThruOperand);
  return Passed;
}

template <typename T, TestFeatures Features> bool testACC(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  properties Align16Props{alignment<16>};
  constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
  properties MinReqAlignProps{alignment<RequiredAlignment>};

  bool Passed = true;

  // Test block_load() that is available on Gen12, DG2 and PVC:
  // 1, 2, 4 or 8  16-byte loads.
  constexpr int NElemsInOword = 16 / sizeof(T);
  Passed &= testACC<T, NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);
  Passed &=
      testACC<T, 2 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 1, 4, Align16Props);
  Passed &=
      testACC<T, 4 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 2, 4, MinReqAlignProps);
  Passed &=
      testACC<T, 8 * NElemsInOword, !CheckMask, !CheckMerge, CheckProperties>(
          Q, 2, 4, Align16Props);

  // Test block_load() without passing compile-time properties argument.
  Passed &=
      testACC<T, NElemsInOword, !CheckMask, !CheckMerge, !CheckProperties>(
          Q, 2, 4, Align16Props);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    // Using mask or cache hints adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.

    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    properties DG2OrPVCProps{cache_hint_L1<cache_hint::streaming>,
                             cache_hint_L2<cache_hint::cached>,
                             alignment<RequiredAlignment>};

    // Test block_load() that is available on DG2/PVC:
    // 1, 2, 3, 4, 8, ... N elements (up to 512-bytes).
    Passed &=
        testACC<T, 1 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, MinReqAlignProps);
    Passed &=
        testACC<T, 2 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 1, 4, MinReqAlignProps);
    Passed &=
        testACC<T, 3 * I32Factor, !CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 8, MinReqAlignProps);
    Passed &= testACC<T, 4 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
        Q, 2, 4, DG2OrPVCProps);
    Passed &= testACC<T, 8 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
        Q, 2, 4, MinReqAlignProps);
    Passed &=
        testACC<T, 16 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
            Q, 2, 4, MinReqAlignProps);
    Passed &=
        testACC<T, 32 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);

    // This call (potentially) and the next call (guaranteed) load the biggest
    // load-able chunk, which requires loading with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::streaming>,
                              cache_hint_L2<cache_hint::cached>, alignment<8>};
    if constexpr (Features == TestFeatures::PVC) {
      Passed &=
          testACC<T, 64 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
              Q, 2, 4, PVCAlign8Props);

      if constexpr (sizeof(T) <= 4)
        Passed &=
            testACC<T, 128 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
                Q, 2, 4, PVCAlign8Props);
    }
  } // TestPVCFeatures

  return Passed;
}

template <typename T, uint16_t N, bool UseMask, bool UsePassThruOperand,
          bool CheckProperties, typename LoadPropertiesT>
bool testSLMAcc(queue Q, uint32_t Groups, uint32_t GroupSize,
                LoadPropertiesT LoadProperties) {
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;

  static_assert(!UsePassThruOperand || UseMask, "Cannot merge without mask");

  uint32_t Size = Groups * GroupSize * N;
  std::cout << "SLM ACC case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask
            << ",UsePassThruOperand=" << UsePassThruOperand;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  constexpr size_t Alignment = getAlignment<T, N, UseMask>(LoadProperties);
  shared_vector In(Size, shared_allocator{Q});
  shared_vector Out(Size, shared_allocator{Q});
  for (int i = 0; i < Size; i++) {
    In[i] = esimd_test::getRandomValue<T>();
    Out[i] = 0;
  }

  try {
    Q.submit([&](handler &CGH) {
       local_accessor<T, 1> LocalAcc(GroupSize * N, CGH);
       auto InPtr = In.data();
       auto OutPtr = Out.data();

       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = NDI.get_global_id(0);
         uint16_t LocalID = NDI.get_local_id(0);
         uint32_t GroupID = NDI.get_group(0);
         uint32_t LocalElemOffset = LocalID * N;
         uint32_t GlobalElemOffset = GlobalID * N;
         if (LocalID == 0)
           for (int I = 0; I < GroupSize * N; I++)
             LocalAcc[I] = InPtr[GlobalID * N + I];
         barrier();

         simd<T, N> Vals;
         simd_mask<1> Mask = (GlobalID + 1) & 0x1;
         // TODO: these 2 lines work-around the problem with scalar
         // conversions to bfloat16. It could be just: "simd<T, N>
         // PassThru(GlobalElemOffset, 1);"
         simd<uint32_t, N> PassThruInt(GlobalElemOffset, 1);
         simd<T, N> PassThru = PassThruInt;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
             if (LocalElemOffset == 0) // try the variant without byte-offset
               Vals = block_load<T, N>(LocalAcc, Mask, PassThru);
             else
               Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T),
                                       Mask, PassThru);
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (LocalElemOffset == 0)
                 Vals = block_load<T, N>(LocalAcc, Mask);
               else
                 Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T),
                                         Mask);
             } else { // if !UseMask
               if (LocalElemOffset == 0)
                 Vals = block_load<T, N>(LocalAcc);
               else
                 Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T));
             }
           }
         } else { // if CheckProperties
           if constexpr (UsePassThruOperand) {
             if (LocalElemOffset == 0)
               Vals = block_load<T, N>(LocalAcc, Mask, PassThru,
                                       LoadPropertiesT{});
             else
               Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T),
                                       Mask, PassThru, LoadPropertiesT{});
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               if (LocalElemOffset == 0)
                 Vals = block_load<T, N>(LocalAcc, Mask, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T),
                                         Mask, LoadPropertiesT{});
             } else { // if !UseMask
               if (LocalElemOffset == 0)
                 Vals = block_load<T, N>(LocalAcc, LoadPropertiesT{});
               else
                 Vals = block_load<T, N>(LocalAcc, LocalElemOffset * sizeof(T),
                                         LoadPropertiesT{});
             }
           }
         }
         Vals.copy_to(OutPtr + GlobalID * N);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed =
      verify<T>(In.data(), Out.data(), Size, N, UseMask, UsePassThruOperand);
  return Passed;
}

template <typename T, TestFeatures Features> bool testSLMAcc(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  bool Passed = true;

  // Test block_load() from SLM that doesn't use the mask is implemented
  // for any N > 1.
  // Ensure that for every call of block_load(local_accessor, offset, ...)
  // the 'alignment' property is specified correctly.
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};
  Passed &= testSLMAcc<T, 1, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testSLMAcc<T, 2, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 1, 4, AlignElemProps);
  Passed &= testSLMAcc<T, 4, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testSLMAcc<T, 8, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);
  Passed &= testSLMAcc<T, 16, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);
  Passed &= testSLMAcc<T, 32, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);
  Passed &= testSLMAcc<T, 64, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, Align16Props);

  // Test block_load() without passing compile-time properties argument.
  Passed &= testSLMAcc<T, 16, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, 4, Align16Props);

  // Test N that is not power of 2, which definitely would require element-size
  // alignment - it works even for byte- and word-vectors if mask is not used.
  // Alignment that is smaller than 16-bytes is not assumed/expected by default
  // and requires explicit passing of the esimd::alignment property.
  Passed &= testSLMAcc<T, 3, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, 4, AlignElemProps);

  // These test case compute wrong values for for the few last elements
  // if the driver is not new enough.
  // TODO: windows version with the fix is not known. Enable it eventually.
  if (sizeof(T) > 2 ||
      esimd_test::isGPUDriverGE(Q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                "27556", "win.just.skip.test", false)) {
    Passed &= testSLMAcc<T, 17, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, 4, AlignElemProps);

    Passed &= testSLMAcc<T, 113, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, 4, AlignElemProps);
  }

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {

    // Using the mask adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.

    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    constexpr size_t ReqiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{alignment<ReqiredAlignment>};

    // Test block_load() that is available on DG2/PVC:
    // 1, 2, 3, 4, 8, ... N elements (up to 512-bytes).
    Passed &=
        testSLMAcc<T, 1 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 2 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
            Q, 1, 4, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 3 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 8, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 4 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 8 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 16 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    Passed &=
        testSLMAcc<T, 32 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, 4, DG2OrPVCProps);
    if constexpr (Features == TestFeatures::PVC) {
      Passed &=
          testSLMAcc<T, 64 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
              Q, 2, 4, DG2OrPVCProps);

      if constexpr (sizeof(T) <= 4)
        Passed &= testSLMAcc<T, 128 * I32Factor, CheckMask, CheckMerge,
                             CheckProperties>(Q, 2, 4, Align16Props);
    }
  } // TestPVCFeatures

  return Passed;
}

template <typename T, uint16_t N, bool UseMask, bool UsePassThruOperand,
          bool CheckProperties, typename LoadPropertiesT>
bool testSLM(queue Q, uint32_t Groups, LoadPropertiesT LoadProperties) {
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;

  constexpr int GroupSize = 8;

  static_assert(!UsePassThruOperand || UseMask, "Cannot merge without mask");

  uint32_t Size = Groups * GroupSize * N;
  std::cout << "SLM case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask
            << ",UsePassThruOperand=" << UsePassThruOperand;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  constexpr size_t Alignment = getAlignment<T, N, UseMask>(LoadProperties);
  // alloc a bit more space (+15 elements) to be able to block read/write using
  // 16-element chunks in initialization/aux code.
  shared_vector In(Size + 15, shared_allocator{Q});
  shared_vector Out(Size + 15, shared_allocator{Q});
  for (int i = 0; i < Size; i++) {
    In[i] = esimd_test::getRandomValue<T>();
    Out[i] = 0;
  }

  try {
    Q.submit([&](handler &CGH) {
       auto InPtr = In.data();
       auto OutPtr = Out.data();

       CGH.parallel_for(Range, [=](sycl::nd_item<1> NDI) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = NDI.get_global_id(0);
         uint16_t LocalID = NDI.get_local_id(0);
         uint32_t GroupID = NDI.get_group(0);
         uint32_t LocalElemOffset = LocalID * N;
         uint32_t GlobalElemOffset = GlobalID * N;

         // Allocate a bit more to safely initialize it with 4-element chunks.
         constexpr uint32_t SLMSize = (GroupSize * N + 4) * sizeof(T);
         slm_init<SLMSize>();

         if (LocalID == 0) {
           for (int I = 0; I < GroupSize * N; I += 4) {
             simd<T, 4> InVec(InPtr + GlobalID * N + I);
             slm_block_store(I * sizeof(T), InVec);
           }
         }
         barrier();

         simd<T, N> Vals;
         simd_mask<1> Mask = (GlobalID + 1) & 0x1;
         // TODO: these 2 lines work-around the problem with scalar
         // conversions to bfloat16. It could be just: "simd<T, N>
         // PassThru(GlobalElemOffset, 1);"
         simd<uint32_t, N> PassThruInt(GlobalElemOffset, 1);
         simd<T, N> PassThru = PassThruInt;
         if constexpr (!CheckProperties) {
           if constexpr (UsePassThruOperand) {
             Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T), Mask,
                                         PassThru);
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T), Mask);
             } else { // if !UseMask
               Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T));
             }
           }
         } else { // if CheckProperties
           if constexpr (UsePassThruOperand) {
             Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T), Mask,
                                         PassThru, LoadPropertiesT{});
           } else { // if !UsePassThruOperand
             if constexpr (UseMask) {
               Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T), Mask,
                                           LoadPropertiesT{});
             } else { // if !UseMask
               Vals = slm_block_load<T, N>(LocalElemOffset * sizeof(T),
                                           LoadPropertiesT{});
             }
           }
         }
         Vals.copy_to(OutPtr + GlobalID * N);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed =
      verify<T>(In.data(), Out.data(), Size, N, UseMask, UsePassThruOperand);
  return Passed;
}

template <typename T, TestFeatures Features> bool testSLM(queue Q) {
  constexpr bool CheckMerge = true;
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;

  bool Passed = true;

  // Test slm_block_load() from SLM that doesn't use the mask is implemented
  // for any N > 1.
  // Ensure that for every call of slm_block_load(offset, ...)
  // the 'alignment' property is specified correctly.
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};
  Passed &= testSLM<T, 1, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, AlignElemProps);
  Passed &= testSLM<T, 2, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 1, AlignElemProps);
  Passed &= testSLM<T, 4, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, AlignElemProps);
  Passed &= testSLM<T, 8, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, AlignElemProps);
  Passed &= testSLM<T, 16, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, Align16Props);
  Passed &= testSLM<T, 32, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, Align16Props);
  Passed &= testSLM<T, 64, !CheckMask, !CheckMerge, CheckProperties>(
      Q, 2, Align16Props);

  // Test block_load() without passing compile-time properties argument.
  Passed &= testSLM<T, 16, !CheckMask, !CheckMerge, !CheckProperties>(
      Q, 2, Align16Props);

  // Test N that is not power of 2, which definitely would require element-size
  // alignment - it works even for byte- and word-vectors if mask is not used.
  // Alignment that is smaller than 16-bytes is not assumed/expected by default
  // and requires explicit passing of the esimd::alignment property.
  //
  // These test case may compute wrong values for some of elements
  // if the driver is not new enough.
  if (esimd_test::isGPUDriverGE(Q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                "27556", "win.just.skip.test", false)) {
    Passed &= testSLM<T, 3, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, AlignElemProps);

    Passed &= testSLM<T, 17, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, AlignElemProps);

    Passed &= testSLM<T, 113, !CheckMask, !CheckMerge, CheckProperties>(
        Q, 2, AlignElemProps);
  }

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    // Using the mask adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires power-or-two elements and
    // the number of bytes loaded per call must not exceed 512.

    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{alignment<RequiredAlignment>};

    // Test block_load() that is available on DG2/PVC:
    // 1, 2, 3, 4, 8, ... N elements (up to 512-bytes).
    Passed &=
        testSLM<T, 1 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, DG2OrPVCProps);
    Passed &= testSLM<T, 2 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
        Q, 1, DG2OrPVCProps);
    Passed &=
        testSLM<T, 3 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, DG2OrPVCProps);
    Passed &= testSLM<T, 4 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
        Q, 2, DG2OrPVCProps);
    Passed &=
        testSLM<T, 8 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, DG2OrPVCProps);
    Passed &=
        testSLM<T, 16 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
            Q, 2, DG2OrPVCProps);
    Passed &=
        testSLM<T, 32 * I32Factor, CheckMask, !CheckMerge, CheckProperties>(
            Q, 2, DG2OrPVCProps);
    if constexpr (Features == TestFeatures::PVC) {
      Passed &=
          testSLM<T, 64 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
              Q, 2, DG2OrPVCProps);

      if constexpr (sizeof(T) <= 4)
        Passed &=
            testSLM<T, 128 * I32Factor, CheckMask, CheckMerge, CheckProperties>(
                Q, 2, Align16Props);
    }
  } // TestPVCFeatures

  return Passed;
}
