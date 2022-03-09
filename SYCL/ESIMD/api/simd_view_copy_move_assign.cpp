//==----- simd_view_copy_move_assign.cpp  - DPC++ ESIMD on-device test -----==//
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

// UNSUPPORTED: windows
// Temprorary disabled on Windows until intel/llvm-test-suite#664 fixed

// This test checks the behavior of simd_view constructors
// and assignment operators.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

template <unsigned VL, class T, class F>
bool test(queue q, std::string str, F funcUnderTest) {
  std::cout << "Testing " << str << ", VL = " << VL << " ...\n";
  T A[VL];
  T B[VL];
  constexpr unsigned HalfVL = VL > 1 ? (VL / 2) : 1;

  // The expected result gets the first half of values from B,
  int gold[VL];
  for (int i = 0; i < VL; ++i) {
    A[i] = -i - 1;
    B[i] = i + 1;
    gold[i] = ((VL > 1) && (i < HalfVL)) ? B[i] : A[i];
  }

  try {
    buffer<T, 1> bufA(A, range<1>(VL));
    buffer<T, 1> bufB(B, range<1>(VL));
    range<1> glob_range{1};

    q.submit([&](handler &cgh) {
       auto PA = bufA.template get_access<access::mode::read_write>(cgh);
       auto PB = bufB.template get_access<access::mode::read>(cgh);
       cgh.parallel_for(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;
         unsigned int offset = i * VL * sizeof(T);
         simd<T, VL> va;
         simd<T, VL> vb;
         if constexpr (VL == 1) {
           va[0] = scalar_load<T>(PA, 0);
           vb[0] = scalar_load<T>(PB, 0);
         } else {
           va.copy_from(PA, offset);
           vb.copy_from(PB, offset);
         }

         auto va_view = va.template select<HalfVL, 1>(0);
         auto vb_view = vb.template select<HalfVL, 1>(0);
         funcUnderTest(va_view, vb_view);

         if constexpr (VL == 1) {
           scalar_store(PB, 0, (T)va[0]);
         } else {
           va.copy_to(PA, offset);
         }
       });
     }).wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < VL; ++i) {
    if (A[i] != gold[i]) {
      err_cnt++;
      std::cout << "failed at index " << i << ": " << A[i] << " != " << gold[i]
                << " (gold)\n";
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (VL - err_cnt) << "/" << VL << ")\n";
  }

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

template <class T, class FuncT>
bool tests(queue &Q, std::string Str, FuncT FuncUnderTest) {
  bool Passed = true;
  Passed &= test<8, T>(Q, Str, FuncUnderTest);
  Passed &= test<1, T>(Q, Str, FuncUnderTest);
  return Passed;
}

template <class T> bool testT(queue &q) {
  std::cout << "--- Testing element type " << typeid(T).name() << " ...\n";
  bool passed = true;
  // copy constructor creates the same view of the underlying data.
  passed &= tests<T>(q, "copy constructor",
                     [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                       auto va_view_copy(va_view);
                       auto vb_view_copy(vb_view);
                       va_view_copy = vb_view_copy;
                     });
  // move constructor transfers the same view of the underlying data.
  passed &= tests<T>(q, "move constructor",
                     [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                       auto va_view_move(std::move(va_view));
                       auto vb_view_move(std::move(vb_view));
                       va_view_move = vb_view_move;
                     });
  // assignment operator copies the underlying data.
  passed &= tests<T>(q, "assignment operator",
                     [](auto &va_view, auto &vb_view)
                         SYCL_ESIMD_FUNCTION { va_view = vb_view; });
  // move assignment operator copies the underlying data.
  passed &= tests<T>(q, "move assignment operator",
                     [](auto &va_view, auto &vb_view)
                         SYCL_ESIMD_FUNCTION { va_view = std::move(vb_view); });

  // construct complete view of a vector.
#if 0
  // TODO: When/if template arguments deducing is implemented for simd
  // constructor accepting simd_view are implemented, the following
  // shorter and more elegant version of the code may be used.
  passed &= tests<T>(q, "constructor from simd",
                  [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                    simd vb = vb_view;
                    simd_view new_vb_view = vb; // ctor from simd
                    va_view = new_vb_view;
                  });
#else
  passed &= test<8, T>(q, "constructor from simd",
                       [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                         simd<int, 4> vb = vb_view;
                         simd_view new_vb_view = vb; // ctor from simd
                         va_view = new_vb_view;
                       });
  passed &= test<1, T>(q, "constructor from simd",
                       [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                         simd<int, 1> vb = vb_view;
                         simd_view new_vb_view = vb; // ctor from simd
                         va_view = new_vb_view;
                       });
#endif

  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;
  passed &= testT<char>(q);
  passed &= testT<short>(q);
  passed &= testT<int>(q);
  passed &= testT<float>(q);
  passed &= testT<half>(q);

  return passed ? 0 : 1;
}
