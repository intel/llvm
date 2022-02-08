//==--------------------------- printf.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// CUDA and HIP don't support printf.
// TODO: esimd_emulator fails due to unimplemented 'single_task()' method
// XFAIL: esimd_emulator
//
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
//
//===----------------------------------------------------------------------===//
//
// The test checks that ESIMD kernels support printf functionality.
// Currently vector and pointer arguments are not supported.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <cstdint>
#include <iostream>

using namespace cl::sycl::ext;

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

static const CONSTANT char format_vec[] = "%d,%d,%d,%d\n";

const CONSTANT char format_hello_world_2[] = "%lu: Hello, World!\n";

int main() {
  {
    queue Queue(esimd_test::ESIMDSelector{},
                esimd_test::createExceptionHandler());

    Queue.submit([&](handler &CGH) {
      CGH.single_task<class integral>([=]() {
        // String
        oneapi::experimental::printf(format_hello_world);
        // Due to a bug in Intel CPU Runtime for OpenCL on Windows, information
        // printed using such format strings (without %-specifiers) might
        // appear in different order if output is redirected to a file or
        // another app
        // FIXME: strictly check output order once the bug is fixed
        // CHECK: {{(Hello, World!)?}}

        // Integral types
        oneapi::experimental::printf(format_int, (int32_t)123);
        oneapi::experimental::printf(format_int, (int32_t)-123);
        // CHECK: 123
        // CHECK-NEXT: -123

        // Floating point types
        {
          // You can declare format string in non-global scope, but in this case
          // static keyword is required
          static const CONSTANT char format[] = "%f\n";
          oneapi::experimental::printf(format, 33.4f);
          oneapi::experimental::printf(format, -33.4f);
        }
        // CHECK-NEXT: 33.4
        // CHECK-NEXT: -33.4

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

  {
    queue Queue(esimd_test::ESIMDSelector{},
                esimd_test::createExceptionHandler());
    // printf in parallel_for
    Queue.submit([&](handler &CGH) {
      CGH.parallel_for<class stream_string>(range<1>(10), [=](id<1> i) {
        // cast to uint64_t to be sure that we pass 64-bit unsigned value
        oneapi::experimental::printf(format_hello_world_2, (uint64_t)i.get(0));
      });
    });
    Queue.wait();
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
    // CHECK-NEXT: {{[0-9]+}}: Hello, World!
  }

  // FIXME: strictly check output order once the bug mentioned above is fixed
  // CHECK: {{(Hello, World!)?}}

  return 0;
}
