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

// Returns true iff verification is passed.
template <typename T>
bool verify(T OutVal, const T *Out, size_t Size, int N, bool UseMask) {
  bool Passed = true;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;

  for (int i = 0; i < Size; i++) {
    bool IsMaskSet = (i / N + 1) & 0x1;
    Tuint Expected = sycl::bit_cast<Tuint>(OutVal);
    if (!UseMask || IsMaskSet)
      Expected = sycl::bit_cast<Tuint>((T)(i + 6));
    Tuint Computed = sycl::bit_cast<Tuint>(Out[i]);
    if (Computed != Expected) {
      Passed = false;
      std::cout << "Out[" << i << "] = " << std::to_string(Computed) << " vs "
                << std::to_string(Expected) << std::endl;
    }
  }
  return Passed;
}

template <typename T, uint16_t N, bool UseMask, bool UseProperties,
          typename StorePropertiesT>
bool testUSM(queue Q, uint32_t Groups, uint32_t Threads,
             StorePropertiesT StoreProperties) {

  uint16_t Size = Groups * Threads * N;

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

  bool Passed = verify(Out_val, Out, Size, N, UseMask);

  sycl::free(Out, Q);

  return Passed;
}

template <typename T, uint16_t N, bool UseMask, bool UseProperties,
          typename StorePropertiesT>
bool testACC(queue Q, uint32_t Groups, uint32_t Threads,
             StorePropertiesT StoreProperties) {

  uint16_t Size = Groups * Threads * N;
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;

  std::cout << "ACC case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseMask=" << UseMask << ",UseProperties=" << UseProperties
            << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};
  constexpr size_t Alignment = getAlignment<T, N, UseMask>(StoreProperties);
  shared_vector Out(Size, shared_allocator{Q});
  T Out_val = esimd_test::getRandomValue<T>();
  for (int i = 0; i < Size; i++)
    Out[i] = Out_val;

  try {
    buffer<T, 1> OutBuf(Out);
    Q.submit([&](handler &cgh) {
       accessor OutAcc{OutBuf, cgh};
       cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = ndi.get_global_id(0);
         uint32_t ElemOff = GlobalID * N;
         simd<T, N> Vals(ElemOff, 1);
         if constexpr (UseMask) {
           simd_mask<1> Mask = (GlobalID + 1) & 0x1;
           if (ElemOff == 0)
             block_store(OutAcc, Vals, Mask, StorePropertiesT{});
           else
             block_store(OutAcc, ElemOff * sizeof(T), Vals, Mask,
                         StorePropertiesT{});
           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 1;
           block_store(OutAcc, ElemOff * sizeof(T), Vals, Mask,
                       StorePropertiesT{});
           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 2;
           auto View = Vals.template select<N, 1>();
           block_store<T, N>(OutAcc, ElemOff * sizeof(T), View, Mask,
                             StorePropertiesT{});
           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 3;
           View = Vals.template select<N, 1>();
           block_store<T, N>(OutAcc, ElemOff * sizeof(T), View, Mask,
                             StorePropertiesT{});
         } else {
           if constexpr (UseProperties)
             block_store(OutAcc, ElemOff * sizeof(T), Vals, StorePropertiesT{});

           else
             block_store(OutAcc, ElemOff * sizeof(T), Vals);

           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 1;
           if constexpr (UseProperties)
             block_store(OutAcc, ElemOff * sizeof(T), Vals, StorePropertiesT{});
           else
             block_store(OutAcc, ElemOff * sizeof(T), Vals);

           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 2;
           auto View = Vals.template select<N, 1>();
           if constexpr (UseProperties)
             block_store<T, N>(OutAcc, ElemOff * sizeof(T), View,
                               StorePropertiesT{});
           else
             block_store<T, N>(OutAcc, ElemOff * sizeof(T), View);

           Vals = block_load<T, N>(OutAcc, ElemOff * sizeof(T));
           Vals += 3;
           View = Vals.template select<N, 1>();
           if constexpr (UseProperties)
             block_store<T, N>(OutAcc, ElemOff * sizeof(T), View,
                               StorePropertiesT{});
           else
             block_store<T, N>(OutAcc, ElemOff * sizeof(T), View);
         }
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed = verify(Out_val, Out.data(), Size, N, UseMask);

  return Passed;
}

template <typename T, bool TestPVCFeatures> bool test_block_store_usm(queue Q) {
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;

  // Test block_store() that is available on Gen12 and PVC.
  Passed &= testUSM<T, 1, !CheckMask, CheckProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 2, !CheckMask, CheckProperties>(Q, 1, 4, AlignElemProps);
  Passed &= testUSM<T, 3, !CheckMask, CheckProperties>(Q, 2, 8, AlignElemProps);
  Passed &= testUSM<T, 4, !CheckMask, CheckProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 8, !CheckMask, CheckProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 16, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);
  // Intentionally check non-power-of-2 simd size - it must work.
  Passed &=
      testUSM<T, 33, !CheckMask, CheckProperties>(Q, 2, 4, AlignElemProps);
  // This test case computes wrong values for for the few last elements
  // if the driver is not new enough.
  // TODO: windows version with the fix is not known. Enable it eventually.
  if (sizeof(T) > 2 ||
      esimd_test::isGPUDriverGE(Q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                "27556", "win.just.skip.test", false))
    Passed &=
        testUSM<T, 67, !CheckMask, CheckProperties>(Q, 1, 4, AlignElemProps);
  // Intentionally check big simd size - it must work.
  Passed &=
      testUSM<T, 128, !CheckMask, CheckProperties>(Q, 2, 4, AlignElemProps);
  Passed &=
      testUSM<T, 256, !CheckMask, CheckProperties>(Q, 1, 4, AlignElemProps);

  // Test block_store() without passing compile-time properties argument.
  Passed &= testUSM<T, 16, !CheckMask, !CheckProperties>(Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, !CheckMask, !CheckProperties>(Q, 2, 4, Align16Props);

  if constexpr (TestPVCFeatures) {
    // Using cache hints adds the requirement to run tests on PVC.
    // Also, PVC variant currently requires a) power-or-two elements,
    // b) the number of bytes stored per call must not exceed 512,
    // c) the alignment of USM ptr + offset to be 4 or 8-bytes(for 8-byte
    // element vectors).
    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties PVCProps{cache_hint_L1<cache_hint::write_back>,
                        cache_hint_L2<cache_hint::write_back>,
                        alignment<RequiredAlignment>};
    // Only d/q-words are supported now.
    // Thus we use this I32Factor for testing purposes and convenience.
    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);

    Passed &= testUSM<T, 1 * I32Factor, !CheckMask, CheckProperties>(Q, 2, 4,
                                                                     PVCProps);
    Passed &= testUSM<T, 2 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testUSM<T, 4 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testUSM<T, 8 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testUSM<T, 16 * I32Factor, CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testUSM<T, 32 * I32Factor, !CheckMask, CheckProperties>(Q, 2, 4,
                                                                      PVCProps);

    // This call (potentially) and the next call (guaranteed) store the biggest
    // store-able chunk, which requires storing with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::write_back>,
                              cache_hint_L2<cache_hint::write_back>,
                              alignment<8>};
    Passed &= testUSM<T, 64 * I32Factor, !CheckMask, CheckProperties>(
        Q, 7, 1, PVCAlign8Props);
    if constexpr (sizeof(T) <= 4)
      Passed &= testUSM<T, 128 * I32Factor, CheckMask, CheckProperties>(
          Q, 1, 4, PVCAlign8Props);

  } // TestPVCFeatures

  return Passed;
}

template <typename T, bool TestPVCFeatures> bool test_block_store_acc(queue Q) {
  constexpr bool CheckMask = true;
  constexpr bool CheckProperties = true;
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;

  // Test block_store() that is available on Gen12 and PVC.

  if constexpr (sizeof(T) >= 4)
    Passed &= testACC<T, 4, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) >= 2)
    Passed &= testACC<T, 8, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);
  Passed &= testACC<T, 16, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) <= 4)
    Passed &=
        testACC<T, 32, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);

  // Intentionally check big simd size - it must work.
  if constexpr (sizeof(T) == 1)
    Passed &=
        testACC<T, 128, !CheckMask, CheckProperties>(Q, 2, 4, Align16Props);

  // Test block_store() without passing compile-time properties argument.
  Passed &= testACC<T, 16, !CheckMask, !CheckProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) <= 4)
    Passed &=
        testACC<T, 32, !CheckMask, !CheckProperties>(Q, 2, 4, Align16Props);

  if constexpr (TestPVCFeatures) {
    // Using cache hints adds the requirement to run tests on PVC.
    // Also, PVC variant currently requires a) power-or-two elements,
    // b) the number of bytes stored per call must not exceed 512,
    // c) the alignment of USM ptr + offset to be 4 or 8-bytes(for 8-byte
    // element vectors).
    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties PVCProps{cache_hint_L1<cache_hint::write_back>,
                        cache_hint_L2<cache_hint::write_back>,
                        alignment<RequiredAlignment>};
    // Only d/q-words are supported now.
    // Thus we use this I32Factor for testing purposes and convenience.
    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);

    Passed &= testACC<T, 1 * I32Factor, !CheckMask, CheckProperties>(Q, 2, 4,
                                                                     PVCProps);
    Passed &= testACC<T, 2 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testACC<T, 4 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testACC<T, 8 * I32Factor, !CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testACC<T, 16 * I32Factor, CheckMask, CheckProperties>(Q, 5, 5,
                                                                     PVCProps);
    Passed &= testACC<T, 32 * I32Factor, !CheckMask, CheckProperties>(Q, 2, 4,
                                                                      PVCProps);

    // This call (potentially) and the next call (guaranteed) store the biggest
    // store-able chunk, which requires storing with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::write_back>,
                              cache_hint_L2<cache_hint::write_back>,
                              alignment<8>};
    Passed &= testACC<T, 64 * I32Factor, !CheckMask, CheckProperties>(
        Q, 7, 1, PVCAlign8Props);

  } // TestPVCFeatures

  return Passed;
}
