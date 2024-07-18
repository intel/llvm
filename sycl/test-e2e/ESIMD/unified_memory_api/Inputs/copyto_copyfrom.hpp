//==-- copyto_copyfrom.hpp - DPC++ ESIMD on-device test -----------==//
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
template <typename T> bool verify(const T *Out, size_t Size, int N) {
  int NumErrors = 0;
  using Tuint = sycl::_V1::ext::intel::esimd::detail::uint_type_t<sizeof(T)>;
  for (int i = 0; i < Size; i++) {
    Tuint Expected = sycl::bit_cast<Tuint>((T)(i * 2));
    Tuint Computed = sycl::bit_cast<Tuint>(Out[i]);
    if (Computed != Expected) {
      NumErrors++;
      if (NumErrors < 32)
        std::cout << "Out[" << i << "] = " << std::to_string(Computed) << " vs "
                  << std::to_string(Expected) << std::endl;
    }
  }
  return NumErrors == 0;
}

template <typename T, uint16_t N, bool UseProperties, typename PropertiesT>
bool testUSM(queue Q, uint32_t Groups, uint32_t Threads,
             PropertiesT Properties) {

  uint16_t Size = Groups * Threads * N;

  std::cout << "USM case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseProperties=" << UseProperties << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};
  constexpr size_t Alignment = getAlignment<T, N, false>(Properties);
  T *In = sycl::aligned_alloc_shared<T>(Alignment, Size, Q);
  T *Out = sycl::aligned_alloc_shared<T>(Alignment, Size, Q);
  T OutVal = esimd_test::getRandomValue<T>();
  for (int i = 0; i < Size; i++) {
    In[i] = i;
    Out[i] = OutVal;
  }

  try {
    Q.submit([&](handler &cgh) {
       cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = ndi.get_global_id(0);
         uint32_t ElemOff = GlobalID * N;
         simd<T, N> Vals(ElemOff, 1);
         simd<T, N> Input;
         simd<int32_t, N> ByteOffsets(ElemOff * sizeof(T), sizeof(T));
         if constexpr (UseProperties) {
           Vals.copy_to(Out + ElemOff, PropertiesT{});
           Input.copy_from(In + ElemOff, PropertiesT{});
         } else {
           Vals.copy_to(Out + ElemOff);
           Input.copy_from(In + ElemOff);
         }
         if constexpr (__ESIMD_DNS::isPowerOf2(N, 64))
           Vals = gather<T, N>(Out, ByteOffsets);
         Vals += Input;
         constexpr int ChunkSize = sizeof(T) * N < 4 ? 2 : 16;

         if constexpr (UseProperties)
           Vals.template copy_to<ChunkSize>(Out + ElemOff, PropertiesT{});
         else
           Vals.template copy_to<ChunkSize>(Out + ElemOff);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = verify(Out, Size, N);

  sycl::free(Out, Q);

  return Passed;
}

template <typename T, uint16_t N, bool UseProperties, typename PropertiesT>
bool testACC(queue Q, uint32_t Groups, uint32_t Threads,
             PropertiesT Properties) {

  uint16_t Size = Groups * Threads * N;
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;

  std::cout << "ACC case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseProperties=" << UseProperties << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{Threads};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};
  shared_vector In(Size, shared_allocator{Q});
  shared_vector Out(Size, shared_allocator{Q});
  T OutVal = esimd_test::getRandomValue<T>();
  for (int i = 0; i < Size; i++) {
    In[i] = i;
    Out[i] = OutVal;
  }

  try {
    buffer<T, 1> InBuf(In);
    buffer<T, 1> OutBuf(Out);
    Q.submit([&](handler &cgh) {
       accessor InAcc{InBuf, cgh};
       accessor OutAcc{OutBuf, cgh};
       cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = ndi.get_global_id(0);
         uint32_t ElemOff = GlobalID * N;
         simd<T, N> Vals(ElemOff, 1);
         simd<T, N> Input;
         simd<int32_t, N> ByteOffsets(ElemOff * sizeof(T), sizeof(T));
         if constexpr (UseProperties) {
           Vals.copy_to(OutAcc, ElemOff * sizeof(T), PropertiesT{});
           Input.copy_from(InAcc, ElemOff * sizeof(T), PropertiesT{});
         } else {
           Vals.copy_to(OutAcc, ElemOff * sizeof(T));
           Input.copy_from(InAcc, ElemOff * sizeof(T));
         }
         if constexpr (__ESIMD_DNS::isPowerOf2(N, 64))
           Vals = gather<T, N>(OutAcc, ByteOffsets);
         Vals += Input;
         constexpr int ChunkSize = sizeof(T) * N < 4 ? 2 : 16;

         if constexpr (UseProperties)
           Vals.template copy_to<decltype(OutAcc), ChunkSize>(
               OutAcc, ElemOff * sizeof(T), PropertiesT{});
         else
           Vals.template copy_to<decltype(OutAcc), ChunkSize>(
               OutAcc, ElemOff * sizeof(T));
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed = verify(Out.data(), Size, N);

  return Passed;
}

template <typename T, uint16_t N, bool UseProperties, typename PropertiesT>
bool testLocalAccSLM(queue Q, uint32_t Groups, PropertiesT Properties) {
  using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, 16>;
  using shared_vector = std::vector<T, shared_allocator>;
  constexpr uint16_t GroupSize = 8;

  uint32_t Size = Groups * GroupSize * N;

  std::cout << "Local Acc case: T=" << esimd_test::type_name<T>() << ",N=" << N
            << ",UseProperties=" << UseProperties << std::endl;

  sycl::range<1> GlobalRange{Groups};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};
  shared_vector In(Size, shared_allocator{Q});

  shared_vector Out(Size, shared_allocator{Q});
  T OutVal = esimd_test::getRandomValue<T>();
  for (int i = 0; i < Size; i++)
    Out[i] = OutVal;

  try {
    Q.submit([&](handler &CGH) {
       local_accessor<T, 1> LocalAcc(GroupSize * N, CGH);
       auto OutPtr = Out.data();

       CGH.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         uint16_t GlobalID = ndi.get_global_id(0);
         uint16_t LocalID = ndi.get_local_id(0);
         uint32_t LocalElemOffset = LocalID * N * sizeof(T);
         uint32_t ElemOff = GlobalID * N;
         simd<T, N> Vals(ElemOff, 1);
         simd<int32_t, N> ByteOffsets(LocalElemOffset, sizeof(T));
         if (LocalID == 0) {
           for (int I = 0; I < Size; I++) {
             simd<T, 1> InVec(I + ElemOff);
             if (GlobalID % GroupSize)
               scatter(LocalAcc, simd<uint32_t, 1>(I * sizeof(T)), InVec);
             else
               InVec.copy_to(LocalAcc, I * sizeof(T));
           }
         }
         barrier();
         constexpr int ChunkSize = sizeof(T) * N < 4 ? 2 : 16;

         if constexpr (UseProperties) {
           if (GlobalID % GroupSize)
             Vals.template copy_from<decltype(LocalAcc), ChunkSize>(
                 LocalAcc, LocalElemOffset, PropertiesT{});
           else
             Vals = gather<T, N>(LocalAcc, ByteOffsets, PropertiesT{});
         } else {
           if (GlobalID % GroupSize)
             Vals.copy_from(LocalAcc, LocalElemOffset);
           else
             Vals = gather<T, N>(LocalAcc, ByteOffsets);
         }

         Vals *= 2;
         Vals.copy_to(OutPtr + ElemOff);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool Passed = verify(Out.data(), Size, N);

  return Passed;
}

template <typename T, TestFeatures Features>
bool test_copyto_copyfrom_usm(queue Q) {
  constexpr bool UseProperties = true;
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};

  bool Passed = true;

  // Test copyto_copyfrom() that is available on Gen12, DG2 and PVC.
  Passed &= testUSM<T, 1, UseProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 2, UseProperties>(Q, 1, 4, AlignElemProps);
  Passed &= testUSM<T, 3, UseProperties>(Q, 2, 8, AlignElemProps);
  Passed &= testUSM<T, 4, UseProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 8, UseProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 16, UseProperties>(Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, UseProperties>(Q, 2, 4, Align16Props);
  // Intentionally check non-power-of-2 simd size - it must work.
  Passed &= testUSM<T, 33, UseProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 67, UseProperties>(Q, 1, 4, AlignElemProps);
  // Intentionally check big simd size - it must work.
  Passed &= testUSM<T, 128, UseProperties>(Q, 2, 4, AlignElemProps);
  Passed &= testUSM<T, 256, UseProperties>(Q, 1, 4, AlignElemProps);

  // Test copyto_copyfrom() without passing compile-time properties argument.
  Passed &= testUSM<T, 16, !UseProperties>(Q, 2, 4, Align16Props);
  Passed &= testUSM<T, 32, !UseProperties>(Q, 2, 4, Align16Props);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    // Using cache hints adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires a) power-or-two elements,
    // b) the number of bytes stored per call must not exceed 512,
    // c) the alignment of USM ptr + offset to be 4 or 8-bytes(for 8-byte
    // element vectors).
    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{cache_hint_L1<cache_hint::streaming>,
                             cache_hint_L2<cache_hint::uncached>,
                             alignment<RequiredAlignment>};
    // Only d/q-words are supported now.
    // Thus we use this I32Factor for testing purposes and convenience.
    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);

    Passed &= testUSM<T, 1 * I32Factor, UseProperties>(Q, 2, 4, DG2OrPVCProps);
    Passed &= testUSM<T, 2 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testUSM<T, 4 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testUSM<T, 8 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testUSM<T, 16 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testUSM<T, 32 * I32Factor, UseProperties>(Q, 2, 4, DG2OrPVCProps);

    // This call (potentially) and the next call (guaranteed) store the biggest
    // store-able chunk, which requires storing with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::streaming>,
                              cache_hint_L2<cache_hint::uncached>,
                              alignment<8>};
    if constexpr (Features == TestFeatures::PVC) {
      Passed &=
          testUSM<T, 64 * I32Factor, UseProperties>(Q, 7, 1, PVCAlign8Props);
      if constexpr (sizeof(T) <= 4)
        Passed &=
            testUSM<T, 128 * I32Factor, UseProperties>(Q, 1, 4, PVCAlign8Props);
    }

  } // TestPVCFeatures

  return Passed;
}

template <typename T, TestFeatures Features>
bool test_copyto_copyfrom_acc(queue Q) {
  constexpr bool UseProperties = true;
  properties Align16Props{alignment<16>};

  bool Passed = true;

  // Test copyto_copyfrom() that is available on Gen12, DG2 and PVC.
  if constexpr (sizeof(T) >= 4)
    Passed &= testACC<T, 4, UseProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) >= 2)
    Passed &= testACC<T, 8, UseProperties>(Q, 2, 4, Align16Props);
  Passed &= testACC<T, 16, UseProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) <= 4)
    Passed &= testACC<T, 32, UseProperties>(Q, 2, 4, Align16Props);

  // Intentionally check big simd size - it must work.
  if constexpr (sizeof(T) == 1)
    Passed &= testACC<T, 128, UseProperties>(Q, 2, 4, Align16Props);

  // Test copyto_copyfrom() without passing compile-time properties argument.
  Passed &= testACC<T, 16, !UseProperties>(Q, 2, 4, Align16Props);
  if constexpr (sizeof(T) <= 4)
    Passed &= testACC<T, 32, !UseProperties>(Q, 2, 4, Align16Props);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    // Using cache hints adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires a) power-or-two elements,
    // b) the number of bytes stored per call must not exceed 512,
    // c) the alignment of USM ptr + offset to be 4 or 8-bytes(for 8-byte
    // element vectors).
    constexpr size_t RequiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{cache_hint_L1<cache_hint::streaming>,
                             cache_hint_L2<cache_hint::uncached>,
                             alignment<RequiredAlignment>};
    // Only d/q-words are supported now.
    // Thus we use this I32Factor for testing purposes and convenience.
    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);

    Passed &= testACC<T, 1 * I32Factor, UseProperties>(Q, 2, 4, DG2OrPVCProps);
    Passed &= testACC<T, 2 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testACC<T, 4 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testACC<T, 8 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testACC<T, 16 * I32Factor, UseProperties>(Q, 5, 5, DG2OrPVCProps);
    Passed &= testACC<T, 32 * I32Factor, UseProperties>(Q, 2, 4, DG2OrPVCProps);

    // This call (potentially) and the next call (guaranteed) store the biggest
    // store-able chunk, which requires storing with 8-byte elements, which
    // requires the alignment to be 8-bytes or more.
    properties PVCAlign8Props{cache_hint_L1<cache_hint::streaming>,
                              cache_hint_L2<cache_hint::uncached>,
                              alignment<8>};
    if constexpr (Features == TestFeatures::PVC)
      Passed &=
          testACC<T, 64 * I32Factor, UseProperties>(Q, 7, 1, PVCAlign8Props);

  } // TestPVCFeatures

  return Passed;
}

template <typename T, TestFeatures Features>
bool test_copyto_copyfrom_local_acc_slm(queue Q) {
  constexpr bool UseProperties = true;

  bool Passed = true;

  // Test copyto_copyfrom() from SLM that doesn't use the mask is implemented
  // for any N > 1.
  // Ensure that for every call of copyto_copyfrom(local_accessor, offset, ...)
  // the 'alignment' property is specified correctly.
  properties Align16Props{alignment<16>};
  properties AlignElemProps{alignment<sizeof(T)>};
  Passed &= testLocalAccSLM<T, 1, UseProperties>(Q, 2, AlignElemProps);
  Passed &= testLocalAccSLM<T, 2, UseProperties>(Q, 1, AlignElemProps);
  Passed &= testLocalAccSLM<T, 4, UseProperties>(Q, 2, AlignElemProps);
  Passed &= testLocalAccSLM<T, 8, UseProperties>(Q, 2, AlignElemProps);
  Passed &= testLocalAccSLM<T, 16, UseProperties>(Q, 2, Align16Props);

  // Test copyto_copyfrom() without passing compile-time properties argument.
  Passed &= testLocalAccSLM<T, 16, !UseProperties>(Q, 2, Align16Props);

  if constexpr (Features == TestFeatures::PVC ||
                Features == TestFeatures::DG2) {
    Passed &= testLocalAccSLM<T, 32, UseProperties>(Q, 2, Align16Props);
    Passed &= testLocalAccSLM<T, 64, UseProperties>(Q, 2, Align16Props);

    // Test N that is not power of 2, which definitely would require
    // element-size
    // alignment - it works even for byte- and word-vectors if mask is not used.
    // Alignment that is smaller than 16-bytes is not assumed/expected by
    // default
    Passed &= testLocalAccSLM<T, 3, UseProperties>(Q, 2, AlignElemProps);

    Passed &= testLocalAccSLM<T, 17, UseProperties>(Q, 2, AlignElemProps);

    Passed &= testLocalAccSLM<T, 113, UseProperties>(Q, 2, AlignElemProps);

    // Using the mask adds the requirement to run tests on DG2/PVC.
    // Also, DG2/PVC variant currently requires power-or-two elements and
    // the number of bytes stored per call must not exceed 512.

    constexpr int I32Factor =
        std::max(static_cast<int>(sizeof(int) / sizeof(T)), 1);
    constexpr size_t ReqiredAlignment = sizeof(T) <= 4 ? 4 : 8;
    properties DG2OrPVCProps{alignment<ReqiredAlignment>,
                             cache_hint_L1<cache_hint::write_back>,
                             cache_hint_L2<cache_hint::write_back>};

    // Test copyto_copyfrom() that is available on PVC:
    // 1, 2, 3, 4, 8, ... N elements (up to 512-bytes).
    Passed &=
        testLocalAccSLM<T, 1 * I32Factor, UseProperties>(Q, 2, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 2 * I32Factor, UseProperties>(Q, 1, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 3 * I32Factor, UseProperties>(Q, 2, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 4 * I32Factor, UseProperties>(Q, 2, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 8 * I32Factor, UseProperties>(Q, 1, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 16 * I32Factor, UseProperties>(Q, 8, DG2OrPVCProps);
    Passed &=
        testLocalAccSLM<T, 32 * I32Factor, UseProperties>(Q, 2, DG2OrPVCProps);
    if constexpr (Features == TestFeatures::PVC)
      Passed &= testLocalAccSLM<T, 64 * I32Factor, !UseProperties>(
          Q, 2, DG2OrPVCProps);
  } // TestPVCFeatures

  return Passed;
}
