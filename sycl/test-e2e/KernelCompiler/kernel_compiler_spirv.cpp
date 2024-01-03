//==- kernel_compiler_spirv.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc

// RUN: ocloc -spv_only -file %S/Kernels/my_kernel.cl -o %t.spv
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.spv

// Test case for the sycl_ext_oneapi_kernel_compiler_spirv extension. This test
// loads a pre-compiled kernel from a SPIR-V file and runs it.

#include <cassert>
#include <fstream>
#include <sycl/sycl.hpp>

using namespace sycl;

void testSyclKernel(sycl::queue &Q, sycl::kernel Kernel, int multiplier,
                    int added) {
  constexpr int N = 4;
  cl_int InputArray[N] = {0, 1, 2, 3};
  cl_int OutputArray[N] = {};

  sycl::buffer InputBuf(InputArray, sycl::range<1>(N));
  sycl::buffer OutputBuf(OutputArray, sycl::range<1>(N));

  Q.submit([&](sycl::handler &CGH) {
    CGH.set_arg(0, InputBuf.get_access<sycl::access::mode::read>(CGH));
    CGH.set_arg(1, OutputBuf.get_access<sycl::access::mode::write>(CGH));
    CGH.parallel_for(sycl::range<1>{N}, Kernel);
  });

  sycl::host_accessor Out{OutputBuf};
  for (int I = 0; I < N; I++) {
    assert(Out[I] == ((I * multiplier) + added));
  }
}

void testKernelFromSpvFile(std::string file_name) {
  namespace syclex = sycl::ext::oneapi::experimental;

  sycl::queue q;

  // Read the SPIR-V module from disk.
  std::ifstream spv_stream(file_name, std::ios::binary);
  spv_stream.seekg(0, std::ios::end);
  size_t sz = spv_stream.tellg();
  spv_stream.seekg(0);
  std::vector<std::byte> spv(sz);
  spv_stream.read(reinterpret_cast<char *>(spv.data()), sz);

  // Create a kernel bundle from the binary SPIR-V.
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclex::create_kernel_bundle_from_source(
          q.get_context(), syclex::source_language::spirv, spv);

  // Build the SPIR-V module for our device.
  sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
      syclex::build(kb_src);

  // Get a "kernel" object representing the kernel from the SPIR-V module.
  sycl::kernel my_kernel = kb_exe.ext_oneapi_get_kernel("my_kernel");

  // Test the kernel
  auto my_num_args = my_kernel.get_info<sycl::info::kernel::num_args>();
  assert(my_num_args == 2 && "my_kernel should take 2 args");
  testSyclKernel(q, my_kernel, 2, 100);
}

int main(int argc, char **argv) {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_SPIRV
  static_assert(false, "KernelCompiler SPIR-V feature test macro undefined");
#endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  assert(argc == 2 && "Usage: ./%t.out <kernel-spv-file>");
  testKernelFromSpvFile(argv[1]);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
