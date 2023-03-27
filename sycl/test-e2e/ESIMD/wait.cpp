//==---------------- wait.cpp  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -emit-llvm -o %t.comp.ll %s
// RUN: sycl-post-link -ir-output-only -lower-esimd -S %t.comp.ll -o %t.out.ll
// RUN: FileCheck --input-file=%t.out.ll %s

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies support of ext::intel::experimental::esimd::wait().
// The function is basically a NOP. It creates explicit scoreboard dependency
// to avoid code motion across wait() and preserve the value computation even
// if it is unused.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iesimd = sycl::ext::intel::experimental::esimd;

bool test(sycl::queue Q, int IArg = 128) {
  try {
    // Test case 1: check wait() with esimd::simd argument.
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<int, 16> A = IArg;
       simd<int, 16> B = A * A;
       iesimd::wait(B);
       // CHECK: mul <16 x i32>
       // CHECK: llvm.genx.dummy.mov
     }).wait();

    // Test case 2: check wait() with esimd::simd_view argument.
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<int, 16> A = IArg;
       simd<int, 16> B = A * 17;
       auto BView = B.select<8, 2>(0);
       BView += 2;
       iesimd::wait(BView);
       // CHECK: mul <16 x i32>
       // CHECK: add <8 x i32>
       // CHECK: llvm.genx.dummy.mov
     }).wait();

    // Test case 3: check wait() that prevesrves one simd and lets
    // optimize away the other/unused one.
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<uint64_t, 8> A = IArg;
       auto B = A * 17;
       iesimd::wait(B);
       auto C = B * 17;
       // CHECK: mul <8 x i64>
       // CHECK-NOT: add <8 x i64>
       // CHECK: llvm.genx.dummy.mov
       // CHECK-NEXT: ret void
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  return true;
}

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << std::endl;

  bool Passed = true;
  Passed &= test(Q);

  std::cout << (Passed ? "Test Passed\n" : "Test FAILED\n");
  return Passed ? 0 : 1;
}
