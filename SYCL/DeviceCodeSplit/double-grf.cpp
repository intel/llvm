//==----------- double-grf.cpp  - DPC++ SYCL on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test verifies effect of
//   set_kernel_properties(kernel_properties::use_double_grf);
// API call in device code:
// - ESIMD/SYCL splitting happens as usual
// - SYCL module is further split into callgraphs for entry points requesting
//   "double GRF" and callgraphs for entry points which are not
// - SYCL device binary images requesting "double GRF" must be compiled with
//   -ze-opt-large-register-file option

// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// TODO/FIXME: esimd_emulator does not support online compilation that
//             invokes 'piProgramBuild'/'piKernelCreate'
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK,CHECK-WITH-VAR

#include "../helpers.hpp"
#include <iostream>
#include <sycl/ext/intel/experimental/kernel_properties.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental;

bool checkResult(const std::vector<float> &A, int Inc) {
  int err_cnt = 0;
  unsigned Size = A.size();

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] != i + Inc)
      if (++err_cnt < 10)
        std::cerr << "failed at A[" << i << "]: " << A[i] << " != " << i + Inc
                  << "\n";
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    return false;
  }
  return true;
}

// Make the double GRF request from non-inlineable function - compiler should
// mark the caller kernel as "double GRF" anyway.
__attribute__((noinline)) void double_grf_marker() {
  set_kernel_properties(kernel_properties::use_double_grf);
}

int main(void) {
  constexpr unsigned Size = 32;
  constexpr unsigned VL = 16;

  std::vector<float> A(Size);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SYCLKernelSingleGRF>(Size,
                                                  [=](id<1> i) { PA[i] += 2; });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 2)) {
    std::cout << "SingleGRF kernel passed\n";
  } else {
    std::cout << "SingleGRF kernel failed\n";
    return 1;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SYCLKernelDoubleGRF>(Size, [=](id<1> i) {
        double_grf_marker();
        PA[i] += 2;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 4)) {
    std::cout << "DoubleGRF kernel passed\n";
  } else {
    std::cout << "DoubleGRF kernel failed\n";
    return 1;
  }

  return 0;
}

// CHECK-LABEL: ---> piProgramBuild(
// CHECK-NOT: -ze-opt-large-register-file
// CHECK-WITH-VAR: -g
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}SingleGRF
// CHECK: ) ---> pi_result : PI_SUCCESS

// CHECK-LABEL: ---> piProgramBuild(
// CHECK-NO-VAR: -ze-opt-large-register-file
// CHECK-WITH-VAR: -g -ze-opt-large-register-file
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}DoubleGRF
// CHECK: ) ---> pi_result : PI_SUCCESS
