//==----------- sycl_esimd_mix.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is basic test for mixing SYCL and ESIMD kernels in the same source and
// in the same program .

// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO/FIXME: esimd_emulator does not support online compilation that
//             invokes 'piProgramBuild'/'piKernelCreate'
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK,CHECK-NO-VAR
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-g" SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK,CHECK-WITH-VAR

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace cl::sycl;

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

    // We need that many workgroups
    cl::sycl::range<1> GlobalRange{Size};
    // We need that many threads in each group
    cl::sycl::range<1> LocalRange{1};

    queue q(gpu_selector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SyclKernel>(GlobalRange * LocalRange,
                                         [=](id<1> i) { PA[i] = PA[i] + 1; });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
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

    // We need that many workgroups
    cl::sycl::range<1> GlobalRange{Size / VL};
    // We need that many threads in each group
    cl::sycl::range<1> LocalRange{1};

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class EsimdKernel>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va;
            va.copy_from(PA, offset);
            simd<float, VL> vc = va + 1;
            vc.copy_to(PA, offset);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 2)) {
    std::cout << "ESIMD kernel passed\n";
  } else {
    std::cout << "ESIMD kernel failed\n";
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
// CHECK-NO-VAR: -vc-codegen
// CHECK-WITH-VAR: -g -vc-codegen
// CHECK: ) ---> pi_result : PI_SUCCESS
// CHECK-LABEL: ---> piKernelCreate(
// CHECK: <const char *>: {{.*}}EsimdKernel
// CHECK: ) ---> pi_result : PI_SUCCESS
