//==------- accessor_load_store.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to outdated memory intrinsic
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the scalar load/store accessor-based ESIMD
// intrinsics.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace cl::sycl;

template <typename T>
using Acc = accessor<T, 1, access_mode::read_write, access::target::device>;

template <typename T> struct Kernel {
  Acc<T> acc;
  Kernel(Acc<T> acc) : acc(acc) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    uint32_t ii = static_cast<uint32_t>(i.get(0));
    T v = scalar_load<T>(acc, ii * sizeof(T));
    v += ii;
    scalar_store<T>(acc, ii * sizeof(T), v);
  }
};

template <typename T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

template <typename T> bool test(queue q, size_t size) {
  std::cout << "Testing T=" << typeid(T).name() << "...\n";
  T *A = new T[size];

  for (unsigned i = 0; i < size; ++i) {
    A[i] = (T)i;
  }

  try {
    buffer<T, 1> buf(A, range<1>(size));
    range<1> glob_range{size};

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      Kernel<T> kernel(acc);
      cgh.parallel_for(glob_range, kernel);
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < size; ++i) {
    T gold = (T)i + (T)i;

    if (A[i] != gold) {
      if (++err_cnt < 10) {
        using T1 = typename char_to_int<T>::type;
        std::cout << "failed at index " << i << ": " << (T1)A[i]
                  << " != " << (T1)gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  delete[] A;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(int argc, char **argv) {
  // TODO the test fails with 1- and 2-byte types when size is not multiple
  // of 4. Supposed reason - wrapping the memory buffer into image1d_buffer
  size_t size = 128; // 117 - fails for char and short

  if (argc > 1) {
    size = atoi(argv[1]);
    size = size == 0 ? 128 : size;
  }
  std::cout << "Using size=" << size << "\n";
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char>(q, size);
  passed &= test<unsigned char>(q, size);
  passed &= test<short>(q, size);
  passed &= test<unsigned short>(q, size);
  passed &= test<int>(q, size);
  passed &= test<unsigned int>(q, size);
  passed &= test<float>(q, size);
  return passed ? 0 : 1;
}
