//==----- simd_view_copy_move_assign.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This test checks the behavior of simd_view constructors
// and assignment operators.

#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <unsigned VL, class T, class F>
bool test(queue q, std::string str, F funcUnderTest) {
  std::cout << "Testing " << str << ", VL = " << VL << " ...\n";
  constexpr size_t Size = 4 * VL;
  T A[Size];
  T B[Size];
  constexpr unsigned HalfVL = VL > 1 ? (VL / 2) : 1;

  // The expected result gets the first half of values from B,
  int gold[Size];
  for (int i = 0; i < Size; ++i) {
    A[i] = -i - 1;
    B[i] = i + 1;
    gold[i] = ((VL > 1) && ((i % VL) < HalfVL)) ? B[i] : A[i];
  }

  try {
    buffer<T, 1> BufA(A, range<1>(Size));
    buffer<T, 1> BufB(B, range<1>(Size));

    q.submit([&](handler &cgh) {
       auto PA = BufA.template get_access<access::mode::read_write>(cgh);
       auto PB = BufB.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for(range<1>{Size / VL}, [=](id<1> i) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;
         unsigned int offset = i * VL * sizeof(T);
         simd<T, VL> va;
         simd<T, VL> vb;
         if constexpr (VL == 1) {
           va[0] = scalar_load<T>(PA, offset);
           vb[0] = scalar_load<T>(PB, offset);
         } else {
           va.copy_from(PA, offset);
           vb.copy_from(PB, offset);
         }

         auto va_view = va.template select<HalfVL, 1>(0);
         auto vb_view = vb.template select<HalfVL, 1>(0);
         funcUnderTest(va_view, vb_view);

         if constexpr (VL == 1) {
           scalar_store(PB, offset, (T)va[0]);
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
  for (unsigned i = 0; i < Size; ++i) {
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
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  esimd_test::printTestLabel(q);
  bool passed = true;
  passed &= testT<char>(q);
  passed &= testT<float>(q);
  if (dev.has(sycl::aspect::fp16))
    passed &= testT<half>(q);

  return passed ? 0 : 1;
}
