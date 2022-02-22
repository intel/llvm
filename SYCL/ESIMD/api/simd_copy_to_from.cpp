//==----- simd_copy_to_from.cpp  - DPC++ ESIMD simd::copy_to/from test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'half' type
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks simd::copy_from/to methods with alignment flags.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/builtins_esimd.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#ifdef _WIN32
#include <malloc.h>
#endif // _WIN32

#include <sycl/ext/intel/experimental/esimd.hpp>

// Workaround for absense of std::aligned_alloc on Windows.
#ifdef _WIN32
#define aligned_malloc(align, size) _aligned_malloc(size, align)
#define aligned_free(ptr) _aligned_free(ptr)
#else // _WIN32
#define aligned_malloc(align, size) std::aligned_alloc(align, size)
#define aligned_free(ptr) std::free(ptr)
#endif // _WIN32

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T, int N, typename Flags>
bool testUSM(queue &Q, T *Src, T *Dst, unsigned Off, Flags) {
  std::cout << "  Running USM test. T=" << typeid(T).name()
            << ", Flags=" << typeid(Flags).name() << ", N=" << N << "...\n";

  for (int I = 0; I < N; ++I) {
    Src[I + Off] = I + 1;
    Dst[I + Off] = 0;
  }

  try {
    Q.submit([&](handler &CGH) {
       CGH.parallel_for(sycl::range<1>{1}, [=](id<1>) SYCL_ESIMD_KERNEL {
         simd<T, N> Vals;
         Vals.copy_from(Src + Off, Flags{});
         Vals.copy_to(Dst + Off, Flags{});
       });
     }).wait();
  } catch (cl::sycl::exception const &E) {
    std::cout << "ERROR. SYCL exception caught: " << E.what() << std::endl;
    return false;
  }

  unsigned NumErrs = 0;
  for (int I = 0; I < N; ++I)
    if (Dst[I + Off] != Src[I + Off])
      if (++NumErrs <= 10)
        std::cout << "failed at " << I << ": " << Dst[I + Off]
                  << " (Dst) != " << Src[I + Off] << " (Src)\n";

  std::cout << (NumErrs == 0 ? "    Passed\n" : "    FAILED\n");
  return NumErrs == 0;
}

template <typename T, int N, typename Flags>
bool testAcc(queue &Q, T *Src, T *Dst, unsigned Off, Flags) {
  std::cout << "  Running accessor test. T=" << typeid(T).name()
            << ", Flags=" << typeid(Flags).name() << ", N=" << N << "...\n";

  for (int I = 0; I < N; ++I) {
    Src[I + Off] = I + 1;
    Dst[I + Off] = 0;
  }

  try {
    buffer<T, 1> SrcB(Src, range<1>(Off + N));
    buffer<T, 1> DstB(Dst, range<1>(Off + N));

    Q.submit([&](handler &CGH) {
       auto SrcA = SrcB.template get_access<access::mode::read>(CGH);
       auto DstA = DstB.template get_access<access::mode::write>(CGH);

       CGH.parallel_for(sycl::range<1>{1}, [=](id<1>) SYCL_ESIMD_KERNEL {
         simd<T, N> Vals;
         Vals.copy_from(SrcA, Off * sizeof(T), Flags{});
         Vals.copy_to(DstA, Off * sizeof(T), Flags{});
       });
     }).wait();
  } catch (cl::sycl::exception const &E) {
    std::cout << "ERROR. SYCL exception caught: " << E.what() << std::endl;
    return false;
  }

  unsigned NumErrs = 0;
  for (int I = 0; I < N; ++I)
    if (Dst[I + Off] != Src[I + Off])
      if (++NumErrs <= 10)
        std::cout << "failed at " << I << ": " << Dst[I + Off]
                  << " (Dst) != " << Src[I + Off] << " (Src)\n";

  std::cout << (NumErrs == 0 ? "    Passed\n" : "    FAILED\n");
  return NumErrs == 0;
}

template <typename T, int N> bool testUSM(queue &Q) {
  struct Deleter {
    queue Q;
    void operator()(T *Ptr) {
      if (Ptr) {
        sycl::free(Ptr, Q);
      }
    }
  };

  std::unique_ptr<T, Deleter> Src(sycl::aligned_alloc_shared<T>(1024u, 512u, Q),
                                  Deleter{Q});
  std::unique_ptr<T, Deleter> Dst(sycl::aligned_alloc_shared<T>(1024u, 512u, Q),
                                  Deleter{Q});

  constexpr unsigned VecAlignOffset = esimd::detail::getNextPowerOf2<N>();

  bool Pass = true;

  Pass &= testUSM<T, N>(Q, Src.get(), Dst.get(), VecAlignOffset + 1u,
                        element_aligned);
  Pass &=
      testUSM<T, N>(Q, Src.get(), Dst.get(), VecAlignOffset, vector_aligned);
  Pass &= testUSM<T, N>(Q, Src.get(), Dst.get(), 128u / sizeof(T),
                        overaligned<128u>);

  return Pass;
}

template <typename T> bool testUSM(queue &Q) {
  bool Pass = true;

  Pass &= testUSM<T, 1>(Q);
  // TODO some of the sizes are commented out to speed-up testing - this test
  // runs ~160 seconds in CI if all sizes are enabled
  // Pass &= testUSM<T, 2>(Q);
  // Pass &= testUSM<T, 3>(Q);
  // Pass &= testUSM<T, 4>(Q);

  Pass &= testUSM<T, 7>(Q);
  Pass &= testUSM<T, 8>(Q);

  // Pass &= testUSM<T, 15>(Q);
  Pass &= testUSM<T, 16>(Q);

  // Pass &= testUSM<T, 24>(Q);
  Pass &= testUSM<T, 25>(Q);

  // Pass &= testUSM<T, 31>(Q);
  Pass &= testUSM<T, 32>(Q);

  Pass &= testUSM<T, 91>(Q);
  // Pass &= testUSM<T, 92>(Q);

  return Pass;
}

template <typename T, int N> bool testAcc(queue &Q) {
  struct Deleter {
    void operator()(T *Ptr) {
      if (Ptr) {
        aligned_free(Ptr);
      }
    }
  };

  std::unique_ptr<T, Deleter> Src(
      static_cast<T *>(aligned_malloc(1024u, 512u * sizeof(T))), Deleter{});
  std::unique_ptr<T, Deleter> Dst(
      static_cast<T *>(aligned_malloc(1024u, 512u * sizeof(T))), Deleter{});

  constexpr unsigned VecAlignOffset = esimd::detail::getNextPowerOf2<N>();

  bool Pass = true;

  Pass &= testAcc<T, N>(Q, Src.get(), Dst.get(), VecAlignOffset + 1u,
                        element_aligned);
  Pass &=
      testAcc<T, N>(Q, Src.get(), Dst.get(), VecAlignOffset, vector_aligned);
  Pass &= testAcc<T, N>(Q, Src.get(), Dst.get(), 128u / sizeof(T),
                        overaligned<128u>);

  return Pass;
}

template <typename T> bool testAcc(queue &Q) {
  bool Pass = true;

  Pass &= testAcc<T, 1>(Q);
  // Pass &= testAcc<T, 2>(Q);
  Pass &= testAcc<T, 3>(Q);
  // Pass &= testAcc<T, 4>(Q);

  // Pass &= testAcc<T, 7>(Q);
  Pass &= testAcc<T, 8>(Q);

  // Pass &= testAcc<T, 15>(Q);
  Pass &= testAcc<T, 16>(Q);

  // Pass &= testAcc<T, 24>(Q);
  Pass &= testAcc<T, 25>(Q);

  // Pass &= testAcc<T, 31>(Q);
  Pass &= testAcc<T, 32>(Q);

  // Pass &= testAcc<T, 91>(Q);
  Pass &= testAcc<T, 92>(Q);

  return Pass;
}

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";

  bool Pass = true;

#ifdef ESIMD_TESTS_FULL_COVERAGE
  Pass &= testUSM<int8_t>(Q);
  Pass &= testUSM<uint16_t>(Q);
  Pass &= testUSM<int32_t>(Q);
  Pass &= testUSM<uint64_t>(Q);
  Pass &= testUSM<float>(Q);
  Pass &= testUSM<double>(Q);
  Pass &= testUSM<half>(Q);

  Pass &= testAcc<uint8_t>(Q);
  Pass &= testAcc<int16_t>(Q);
  Pass &= testAcc<uint32_t>(Q);
  Pass &= testAcc<int64_t>(Q);
  Pass &= testAcc<float>(Q);
  Pass &= testAcc<double>(Q);
  Pass &= testAcc<half>(Q);
#else
  Pass &= testUSM<uint16_t>(Q);
  Pass &= testUSM<float>(Q);

  Pass &= testAcc<int16_t>(Q);
  Pass &= testAcc<float>(Q);
#endif

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
