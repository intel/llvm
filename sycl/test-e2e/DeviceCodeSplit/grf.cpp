//==----------- double-grf.cpp  - DPC++ SYCL on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test verifies effect of the register_alloc_mode kernel property
// API call in device code:
// - ESIMD/SYCL splitting happens as usual
// - SYCL module is further split into callgraphs for entry points for
//   each value
// - SYCL device binary images are compiled with the corresponding
//   compiler option

// REQUIRES: gpu && gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-WITH-VAR
// RUN: %{build} -DUSE_NEW_API=1 -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-WITH-VAR
// RUN: %{build} -DUSE_AUTO_GRF=1 -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-WITH-VAR
// RUN: %{build} -DUSE_NEW_API=1 -DUSE_AUTO_GRF=1 -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-WITH-VAR
#include "../helpers.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#ifdef USE_NEW_API
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#else
#include <sycl/detail/kernel_properties.hpp>
#endif

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::oneapi::experimental;

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
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

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
#if defined(USE_NEW_API) && defined(USE_AUTO_GRF)
    properties prop{grf_size_automatic};
#elif defined(USE_NEW_API)
    properties prop{grf_size<256>};
#elif USE_AUTO_GRF
    properties prop{register_alloc_mode<register_alloc_mode_enum::automatic>};
#else
    properties prop{register_alloc_mode<register_alloc_mode_enum::large>};
#endif
    queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SYCLKernelSpecifiedGRF>(
          Size, prop, [=](id<1> i) { PA[i] += 2; });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 4)) {
    std::cout << "SpecifiedGRF kernel passed\n";
  } else {
    std::cout << "SpecifiedGRF kernel failed\n";
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
// CHECK-AUTO-NO-VAR: -ze-intel-enable-auto-large-GRF-mode
// CHECK-AUTO-WITH-VAR: -g -ze-intel-enable-auto-large-GRF-mode
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}SpecifiedGRF
// CHECK: ) ---> pi_result : PI_SUCCESS
