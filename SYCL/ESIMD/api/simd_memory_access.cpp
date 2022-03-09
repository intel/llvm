//==------- simd_memory_access.cpp  - DPC++ ESIMD on-device test -----------==//
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
//
// The test checks functionality of the memory access APIs which are members of
// the simd class.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

template <typename T>
using Acc = accessor<T, 1, access_mode::read_write, access::target::device>;

template <typename T, int N, bool IsAcc> struct Kernel;

// Accessor-based kernel.
template <typename T, int N> struct Kernel<T, N, true> {
  Acc<T> acc;
  Kernel(Acc<T> acc) : acc(acc) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    const uint32_t ii = static_cast<uint32_t>(i.get(0));
    simd<T, N> v;
    const auto offset = ii * sizeof(v);
    v.copy_from(acc, offset);
    v += simd<T, N>(ii * N, 1);
    v.copy_to(acc, offset);
  }
};

// Pointer-based kernel.
template <typename T, int N> struct Kernel<T, N, false> {
  T *ptr;
  Kernel(T *ptr) : ptr(ptr) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    const uint32_t ii = static_cast<uint32_t>(i.get(0));
    simd<T, N> v;
    const auto offset = ii * (sizeof(v) / sizeof(T));
    v.copy_from(ptr + offset);
    v += simd<T, N>(ii * N, 1);
    v.copy_to(ptr + offset);
  }
};

template <typename T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

template <bool IsAcc, typename T> void free_mem(T *ptr, sycl::queue q) {
  if constexpr (IsAcc) {
    delete[] ptr;
  } else {
    sycl::free(ptr, q);
  }
}

// The main test routine.
template <typename T, int N, bool IsAcc> bool test(queue q, size_t size) {
  std::cout << "Testing T=" << typeid(T).name() << ", N=" << N
            << " using accessor=" << IsAcc << "...\n";
  T *A;
  if constexpr (IsAcc)
    A = new T[size];
  else
    A = sycl::malloc_shared<T>(size, q);

  for (unsigned i = 0; i < size; ++i) {
    A[i] = i; // should not be zero to test `copy_from` really works
  }

  try {
    if constexpr (IsAcc) {
      buffer<T, 1> buf(A, range<1>(size));
      range<1> glob_range{size / N};

      auto e = q.submit([&](handler &cgh) {
        auto acc = buf.template get_access<access::mode::read_write>(cgh);
        Kernel<T, N, true> kernel(acc);
        cgh.parallel_for(glob_range, kernel);
      });
    } else {
      range<1> glob_range{size / N};

      auto e = q.submit([&](handler &cgh) {
        Kernel<T, N, false> kernel(A);
        cgh.parallel_for(glob_range, kernel);
      });
    }
    q.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free_mem<IsAcc>(A, q);
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < size; ++i) {
    T gold = (T)(i * 2);
    T val = A[i];

    if (val != gold) {
      if (++err_cnt < 10) {
        using T1 = typename char_to_int<T>::type;
        std::cout << "failed at index " << i << ": " << (T1)val
                  << " != " << (T1)gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  free_mem<IsAcc>(A, q);

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(int argc, char **argv) {
  size_t size = 32 * 7;

  if (argc > 1) {
    size = atoi(argv[1]);
    size = size == 0 ? 128 : size;
  }
  if (size % 32 != 0) {
    std::cerr << "*** ERROR: size (" << size << ") must be a multiple of 32\n";
    return 2;
  }
  std::cout << "Using size=" << size << "\n";
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char, 32, true>(q, size);
  passed &= test<unsigned char, 16, true>(q, size);
  passed &= test<short, 32, true>(q, size);
  passed &= test<short, 16, true>(q, size);
  passed &= test<unsigned short, 8, true>(q, size);
  passed &= test<int, 32, true>(q, size);
  passed &= test<unsigned int, 32, true>(q, size);
  passed &= test<float, 32, true>(q, size);
  passed &= test<half, 32, true>(q, size);

  passed &= test<char, 32, false>(q, size);
  passed &= test<unsigned char, 16, false>(q, size);
  passed &= test<short, 32, false>(q, size);
  passed &= test<short, 16, false>(q, size);
  passed &= test<unsigned short, 8, false>(q, size);
  passed &= test<int, 32, false>(q, size);
  passed &= test<unsigned int, 32, false>(q, size);
  passed &= test<float, 32, false>(q, size);
  passed &= test<half, 32, false>(q, size);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
