//==------- replicate_smoke.cpp  - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks main functionality of the esimd::replicate_vs_w_hs function.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <class T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

template <class T> bool verify(T *data_arr, T *gold_arr, int NonZeroN, int N) {
  int err_cnt = 0;

  for (unsigned i = 0; i < NonZeroN; ++i) {
    T val = data_arr[i];
    T gold = gold_arr[i];

    if (val != gold) {
      if (++err_cnt < 10) {
        using T1 = typename char_to_int<T>::type;
        std::cout << "  failed at index " << i << ": " << (T1)val
                  << " != " << (T1)gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(NonZeroN - err_cnt) / (float)NonZeroN) * 100.0f
              << "% (" << (NonZeroN - err_cnt) << "/" << NonZeroN << ")\n";
  }
  for (unsigned i = NonZeroN; i < N; ++i) {
    T val = data_arr[i];
    T gold = (T)0;

    if (val != gold) {
      if (++err_cnt < 10) {
        using T1 = typename char_to_int<T>::type;
        std::cout << "  additional failure at index " << i << ": " << (T1)val
                  << " != " << (T1)gold << " (gold)\n";
      }
    }
  }
  return err_cnt == 0;
}

template <class T> struct DataMgr {
  T *src;
  T *dst;

  DataMgr(int N) {
    src = new T[N];
    dst = new T[N];

    for (int i = 0; i < N; i++) {
      src[i] = (T)i;
      dst[i] = (T)0;
    }
  }

  ~DataMgr() {
    delete[] src;
    delete[] dst;
  }
};

template <class T, int VL, int N, int Rep, int Vs, int W, int Hs>
bool test_impl(queue q, int offset, T (&&gold)[N]) {
  std::cout << "Testing T=" << typeid(T).name() << " Rep=" << Rep << " "
            << "Vs=" << Vs << " "
            << "W=" << W << " "
            << "Hs=" << Hs << " "
            << "Off=" << offset << "...";

  DataMgr<T> dm(VL);

  try {
    sycl::buffer<T, 1> src_buf(dm.src, VL);
    sycl::buffer<T, 1> dst_buf(dm.dst, VL);

    q.submit([&](handler &cgh) {
       auto src_acc = src_buf.template get_access<access::mode::read>(cgh);
       auto dst_acc = dst_buf.template get_access<access::mode::write>(cgh);

       cgh.single_task([=]() SYCL_ESIMD_KERNEL {
         simd<T, VL> src(src_acc, 0);
         simd<T, N> res =
             src.template replicate_vs_w_hs<Rep, Vs, W, Hs>(offset);
         res.copy_to(dst_acc, 0);
       });
     }).wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }
  bool passed = verify<T>(dm.dst, gold, N, VL);
  std::cout << (passed ? "ok" : "FAILED") << "\n";
  return passed;
}

template <class T> bool test(queue q) {
  bool passed = true;
  // 'x' and '.' represent source elements, 'x' represent selected elements
  // Numbers in the result - source's elements ordinals

  // test_impl: <class T, int VL, int N, int Rep, int Vs, int W, int Hs>

  // clang-format off
  //|<----------- Vs=16 ----------->|
  //   v-------v-------v W=3
  // . x . . . x . . . x . . . . . . \ Rep=2
  // . x . . . x . . . x . . . . . . /
  //  |<- Hs=4->|
  // clang-format on
  passed &= test_impl<T, 32, 6, 2, 16, 3, 4>(
      q, 1 /*off*/,
      {// expected result, other elements are zeroes
       (T)1, (T)5, (T)9, (T)17, (T)21, (T)25});

  // clang-format off
  //|<----------- Vs=17 ------------->|
  //   v-------v-------v W=3
  // . x . . . x . . . x . . . . . . . \ Rep=2
  // . x . . . x . . . x . . . . .     /
  //  |<- Hs=4->|
  // clang-format on
  passed &= test_impl<T, 32, 6, 2, 17, 3, 4>(
      q, 1 /*off*/,
      {// expected result, other elements are zeroes
       (T)1, (T)5, (T)9, (T)18, (T)22, (T)26});

  // clang-format off
  // AOS 7x3 => SOA 3x7:
  // x0y0z0x1y1z1x2y2z2x3y3z3x4y4z4x5y5z5x6y6z6
  // =>
  // x0x1x2x3x4x5x6y0y1y2y3y4y5y6z0z1z2z3z4z5z6
  // Rep=3, VS=1, HS=3, W=7
  // clang-format on
  passed &= test_impl<T, 21, 21, 3, 1, 7, 3>(
      q, 0 /*off*/,
      {// expected result, other elements are zeroes
       (T)0, (T)3, (T)6, (T)9,  (T)12, (T)15, (T)18,
       (T)1, (T)4, (T)7, (T)10, (T)13, (T)16, (T)19,
       (T)2, (T)5, (T)8, (T)11, (T)14, (T)17, (T)20});

  // . . . . . . . . . . x . . . . . . . . . . . . . . . . . . . . .
  passed &= test_impl<T, 32, 1, 1, 0, 1, 0>(
      q, 10 /*off*/,
      {
          (T)10 // expected result, other elements are zeroes
      });
  return passed;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  bool passed = true;

  passed &= test<half>(q);
  passed &= test<unsigned char>(q);
  passed &= test<short>(q);
  passed &= test<unsigned short>(q);
  passed &= test<int>(q);
  passed &= test<uint64_t>(q);
  passed &= test<float>(q);
  passed &= test<double>(q);

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
