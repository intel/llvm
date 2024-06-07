//==--------------------------- printf.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// RUN: %{build} -fsycl-device-code-split=per_kernel -D__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ -o %t_var.out
// RUN: %{run} %t_var.out | FileCheck %s
//
//===----------------------------------------------------------------------===//
//
// The test checks that ESIMD kernels support printf functionality.
// Currently vector and pointer arguments are not supported.

#include "esimd_test_utils.hpp"

#include <cstdint>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

using namespace sycl::ext;

// According to OpenCL C spec, the format string must be in constant address
// space
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

// This is one of the possible ways to define a format string in a correct
// address space
static const CONSTANT char format_hello_world[] = "Hello, World!\n";

// Static isn't really needed if you define it in global scope
const CONSTANT char format_int[] = "%d\n";

int main() {
  queue Queue(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  {
    Queue.submit([&](handler &CGH) {
      CGH.single_task([=]() SYCL_ESIMD_KERNEL {
        // String
        oneapi::experimental::printf(format_hello_world);
        // CHECK: Hello, World!

        // Integral types
        oneapi::experimental::printf(format_int, (int32_t)123);
        oneapi::experimental::printf(format_int, (int32_t)-123);
        // CHECK-NEXT: 123
        // CHECK-NEXT: -123

        // String types
        {
          static CONSTANT const char str_arg[] = "foo";
          static const CONSTANT char format[] = "%s\n";
          oneapi::experimental::printf(format, str_arg);
        }
        // CHECK-NEXT: foo
      });
    });
    Queue.wait();
  }

#ifdef __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__
  // Currently printf will promote floating point values to doubles.
  // __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ changes the behavior to use
  // a variadic function, so if it is defined it will promote the floating
  // point arguments.
  if (Queue.get_device().has(sycl::aspect::fp64))
#endif // __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__
  {
    Queue.submit([&](handler &CGH) {
      CGH.single_task<class floating_points>([=]() {
        // Floating point types
        {
          // You can declare format string in non-global scope, but in this case
          // static keyword is required
          static const CONSTANT char format[] = "%.1f\n";
          ext::oneapi::experimental::printf(format, 33.4f);
          ext::oneapi::experimental::printf(format, -33.4f);
        }
      });
    });
    Queue.wait();
  }
#ifdef __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__
  else {
    std::cout << "Skipped floating point test." << std::endl;
    std::cout << "Skipped floating point test." << std::endl;
  }
#endif // __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__
  // CHECK-NEXT: {{(33.4|Skipped floating point test.)}}
  // CHECK-NEXT: {{(-33.4|Skipped floating point test.)}}

  {
    // printf in parallel_for
    constexpr int SIMD_SIZE = 16;
    constexpr int WORK_SIZE = SIMD_SIZE;
    int *Mem = malloc_shared<int>(WORK_SIZE * SIMD_SIZE, Queue);
    for (int I = 0; I < WORK_SIZE * SIMD_SIZE; I++)
      Mem[I] = I;
    std::cout << "Start parallel_for:" << std::endl;
    Queue
        .parallel_for(range<1>(WORK_SIZE),
                      [=](id<1> i) SYCL_ESIMD_KERNEL {
                        static const CONSTANT char STR_LU_D[] =
                            "Thread-id: %d, Value: %d\n";
                        ext::intel::esimd::simd<int, SIMD_SIZE> Vec(
                            Mem + i * SIMD_SIZE);
                        // cast to uint64_t to be sure that we pass 64-bit
                        // unsigned value
                        oneapi::experimental::printf(STR_LU_D, (uint64_t)i[0],
                                                     (int)Vec[i]);
                      })
        .wait();
    free(Mem, Queue);
    // CHECK-LABEL: Start parallel_for
    // CHECK-DAG: Thread-id: 0, Value: 0
    // CHECK-DAG: Thread-id: 1, Value: 17
    // CHECK-DAG: Thread-id: 2, Value: 34
    // CHECK-DAG: Thread-id: 3, Value: 51
    // CHECK-DAG: Thread-id: 4, Value: 68
    // CHECK-DAG: Thread-id: 5, Value: 85
    // CHECK-DAG: Thread-id: 6, Value: 102
    // CHECK-DAG: Thread-id: 7, Value: 119
    // CHECK-DAG: Thread-id: 8, Value: 136
    // CHECK-DAG: Thread-id: 9, Value: 153
    // CHECK-DAG: Thread-id: 10, Value: 170
    // CHECK-DAG: Thread-id: 11, Value: 187
    // CHECK-DAG: Thread-id: 12, Value: 204
    // CHECK-DAG: Thread-id: 13, Value: 221
    // CHECK-DAG: Thread-id: 14, Value: 238
    // CHECK-DAG: Thread-id: 15, Value: 255
  }

  return 0;
}
