// Copyright (C) Intel

//===--- vec_add.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s -S | FileCheck %s

#include <sycl/sycl.hpp>
#define N 32
// clang-format off
// CHECK: func.func private {{.*}}vec_add_device_simple{{.*}}sycl{{.*}}handler{{.*}}
// CHECK-DAG: %[[VEC1:.*]] = affine.load %[[ACCESSOR1:.*]][0] : memref<?xf32>
// CHECK-DAG: %[[VEC2:.*]] = affine.load %[[ACCESSOR2:.*]][0] : memref<?xf32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addf %[[VEC1]], %[[VEC2]] : f32
// CHECK-NEXT: affine.store %[[RESULT]], %[[VEC3:.*]][0] : memref<?xf32>
// CHECK-NEXT: return
// clang-format on

void vec_add_device_simple(std::array<float, N> &VA, std::array<float, N> &VB,
                           std::array<float, N> &VC) {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{N};

  {
    auto bufA = sycl::buffer<float, 1>{VA.data(), range};
    auto bufB = sycl::buffer<float, 1>{VB.data(), range};
    auto bufC = sycl::buffer<float, 1>{VC.data(), range};

    q.submit([&](sycl::handler &cgh) {
      auto A = bufA.get_access<sycl::access::mode::read>(cgh);
      auto B = bufB.get_access<sycl::access::mode::read>(cgh);
      auto C = bufC.get_access<sycl::access::mode::write>(cgh);

      // kernel
      cgh.parallel_for<class vec_add_simple>(
          range, [=](sycl::id<1> id) { C[id] = A[id] + B[id]; });
    });
  }
}

void init(std::array<float, N> &h_a, std::array<float, N> &h_b,
          std::array<float, N> &h_c, std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
    h_c[i] = 0.0f;
    h_r[i] = 0.0f;
  }
}

void vec_add_host(std::array<float, N> &h_a, std::array<float, N> &h_b,
                  std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    h_r[i] = h_a[i] + h_b[i];
  }
}

bool check_result(std::array<float, N> &h_c, std::array<float, N> &h_r) {
  for (int i = 0; i < N; i++) {
    if (h_r[i] != h_c[i]) {
      std::cerr << "Mismatch at element " << i << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  std::array<float, N> h_a;
  std::array<float, N> h_b;
  std::array<float, N> h_c;
  std::array<float, N> h_r; // (result)

  // initialize vectors
  init(h_a, h_b, h_c, h_r);

  vec_add_host(h_a, h_b, h_r);

  vec_add_device_simple(h_a, h_b, h_c);

  if (!check_result(h_c, h_r))
    exit(1);

  std::cout << "Results are correct\n";
  return 0;
}
