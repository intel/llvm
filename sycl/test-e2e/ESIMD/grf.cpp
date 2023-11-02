//==----------- grf.cpp  - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test verifies effect of the register_alloc_mode kernel property
// API call in device code:
// - ESIMD/SYCL splitting happens as usual
// - ESIMD module is further split into callgraphs for entry points for
//   each value
// - ESIMD device binary images are compiled with the corresponding
//   compiler option

// REQUIRES: gpu-intel-pvc
//             invokes 'piProgramBuild'/'piKernelCreate'
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-WITH-VAR
// RUN: %{build} -DUSE_NEW_API=1 -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-WITH-VAR
// RUN: %{build} -DUSE_AUTO -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO-WITH-VAR
#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#if defined(USE_NEW_API) || defined(USE_AUTO)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#else
#include <sycl/detail/kernel_properties.hpp>
#endif

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

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
    queue q(gpu_selector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SyclKernel>(Size,
                                         [=](id<1> i) { PA[i] = PA[i] + 1; });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 1)) {
    std::cout << "SYCL kernel passed\n";
  } else {
    std::cout << "SYCL kernel failed\n";
    return 1;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class EsimdKernel>(Size, [=](id<1> i) SYCL_ESIMD_KERNEL {
        unsigned int offset = i * VL * sizeof(float);
        simd<float, VL> va;
        va.copy_from(PA, offset);
        simd<float, VL> vc = va + 1;
        vc.copy_to(PA, offset);
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 2)) {
    std::cout << "ESIMD kernel passed\n";
  } else {
    std::cout << "ESIMD kernel failed\n";
    return 1;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
#ifdef USE_AUTO
    sycl::ext::oneapi::experimental::properties prop{grf_size_automatic};
#elif defined(USE_NEW_API)
    sycl::ext::oneapi::experimental::properties prop{grf_size<256>};
#else
    sycl::ext::oneapi::experimental::properties prop{
        register_alloc_mode<register_alloc_mode_enum::large>};
#endif
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class EsimdKernelSpecifiedGRF>(
          Size, prop, [=](id<1> i) SYCL_ESIMD_KERNEL {
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va;
            va.copy_from(PA, offset);
            simd<float, VL> vc = va + 1;
            vc.copy_to(PA, offset);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 3)) {
    std::cout << "ESIMD specified GRF kernel passed\n";
  } else {
    std::cout << "ESIMD specified GRF kernel failed\n";
    return 1;
  }

  return 0;
}

// Regular SYCL kernel is compiled without -vc-codegen option

// CHECK-LABEL: ---> piProgramBuild(
// CHECK-NOT: -vc-codegen
// CHECK-WITH-VAR: -g
// CHECK-NOT: -vc-codegen
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}SyclKernel
// CHECK: ) ---> pi_result : PI_SUCCESS

// For ESIMD kernels, -vc-codegen option is always preserved,
// regardless of SYCL_PROGRAM_COMPILE_OPTIONS value.

// CHECK-LABEL: ---> piProgramBuild(
// CHECK-NO-VAR: -vc-codegen -disable-finalizer-msg
// CHECK-WITH-VAR: -g -vc-codegen -disable-finalizer-msg
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}EsimdKernel
// CHECK: ) ---> pi_result : PI_SUCCESS

// Kernels requesting GRF are grouped into separate module and compiled
// with the respective option regardless of SYCL_PROGRAM_COMPILE_OPTIONS value.

// CHECK-LABEL: ---> piProgramBuild(
// CHECK-NO-VAR: -vc-codegen -disable-finalizer-msg -doubleGRF
// CHECK-WITH-VAR: -g -vc-codegen -disable-finalizer-msg -doubleGRF
// CHECK-AUTO-NO-VAR: -vc-codegen -disable-finalizer-msg -ze-intel-enable-auto-large-GRF-mode
// CHECK-AUTO-WITH-VAR: -g -vc-codegen -disable-finalizer-msg -ze-intel-enable-auto-large-GRF-mode
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}EsimdKernelSpecifiedGRF
// CHECK: ) ---> pi_result : PI_SUCCESS
