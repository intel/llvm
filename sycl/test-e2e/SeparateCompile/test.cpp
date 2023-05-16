// UNSUPPORTED: cuda || hip
// CUDA and HIP don't support SPIR-V.
//
// FIXME Disabled fallback assert as it'll require either online linking or
// explicit offline linking step here
// FIXME separate compilation requires -fno-sycl-dead-args-optimization
// >> ---- compile src1
// >> device compilation...
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT -fno-sycl-dead-args-optimization -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_a.h %s -c -o a_kernel.bc -Wno-sycl-strict
// >> host compilation...
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 %include_option sycl_ihdr_a.h %debug_option -c %s -o a.o %sycl_options -fno-sycl-dead-args-optimization -Wno-sycl-strict
//
// >> ---- compile src2
// >> device compilation...
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT -DB_CPP=1 -fno-sycl-dead-args-optimization -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_b.h %s -c -o b_kernel.bc -Wno-sycl-strict
// >> host compilation...
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT -DB_CPP=1 %cxx_std_optionc++17 %include_option sycl_ihdr_b.h %debug_option -c %s -o b.o %sycl_options -fno-sycl-dead-args-optimization -Wno-sycl-strict
//
// >> ---- bundle .o with .spv
// >> run bundler
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -input=a.o -input=a_kernel.bc -output=a_fat.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -input=b.o -input=b_kernel.bc -output=b_fat.o
//
// >> ---- unbundle fat objects
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -output=a.o -output=a_kernel.bc -input=a_fat.o -unbundle
// RUN: clang-offload-bundler -type=o -targets=host-x86_64,sycl-spir64-pc-linux-gnu -output=b.o -output=b_kernel.bc -input=b_fat.o -unbundle
//
// As we are doing a separate device compilation here, we need to explicitly
// add the device lib instrumentation (itt_compiler_wrapper)
// >> ---- unbundle compiler wrapper device object
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=%sycl_static_libs_dir/libsycl-itt-compiler-wrappers%obj_ext -output=compiler_wrappers.bc -unbundle
//
// >> ---- link device code
// RUN: llvm-link -o=app.bc a_kernel.bc b_kernel.bc compiler_wrappers.bc
//
// >> convert linked .bc to spirv
// RUN: llvm-spirv -o=app.spv app.bc
//
// >> ---- wrap device binary
// >> produce .bc
// RUN: clang-offload-wrapper -o wrapper.bc -host=x86_64 -kind=sycl -target=spir64 app.spv
//
// >> compile .bc to .o
// RUN: %clangxx -c wrapper.bc -o wrapper.o
//
// >> ---- link the full hetero app
// RUN: %clangxx wrapper.o a.o b.o -o app.exe %sycl_options
// RUN: %BE_RUN_PLACEHOLDER ./app.exe | FileCheck %s
// CHECK: pass

//==----------- test.cpp - Tests SYCL separate compilation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef B_CPP
// -----------------------------------------------------------------------------
#include <iostream>
#include <sycl/sycl.hpp>

int run_test_b(int v) {
  int arr[] = {v};
  {
    sycl::queue deviceQueue;
    sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_b>([=]() { acc[0] *= 3; });
    });
  }
  return arr[0];
}

#else // !B_CPP

// -----------------------------------------------------------------------------
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;

const int VAL = 10;

extern int run_test_b(int);

int run_test_a(int v) {
  int arr[] = {v};
  {
    sycl::queue deviceQueue;
    sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_a>([=]() { acc[0] *= 2; });
    });
  }
  return arr[0];
}

int main(int argc, char **argv) {
  bool pass = true;

  int test_a = run_test_a(VAL);
  const int GOLD_A = 2 * VAL;

  if (test_a != GOLD_A) {
    std::cout << "FAILD test_a. Expected: " << GOLD_A << ", got: " << test_a
              << "\n";
    pass = false;
  }

  int test_b = run_test_b(VAL);
  const int GOLD_B = 3 * VAL;

  if (test_b != GOLD_B) {
    std::cout << "FAILD test_b. Expected: " << GOLD_B << ", got: " << test_b
              << "\n";
    pass = false;
  }

  if (pass) {
    std::cout << "pass\n";
  }
  return pass ? 0 : 1;
}
#endif // !B_CPP
