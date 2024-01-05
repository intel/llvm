//==- kernel_compiler_spirv.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc

// RUN: %clang -c -target spir64 -O0 -emit-llvm %S/Kernels/spirv_tests.cl -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.spv

// Test case for the sycl_ext_oneapi_kernel_compiler_spirv extension. This test
// loads pre-compiled kernels from a SPIR-V file and runs them.

#include <array>
#include <cassert>
#include <fstream>
#include <string>
#include <sycl/sycl.hpp>

void testSimpleKernel(sycl::queue &q, const sycl::kernel &kernel,
                      int multiplier, int added) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  assert(num_args == 2 && "kernel should take 2 args");

  constexpr int N = 4;
  std::array<int, N> input_array{0, 1, 2, 3};
  std::array<int, N> output_array{};

  sycl::buffer input_buffer(input_array.data(), sycl::range<1>(N));
  sycl::buffer output_buffer(output_array.data(), sycl::range<1>(N));

  q.submit([&](sycl::handler &cgh) {
    cgh.set_arg(0, input_buffer.get_access<sycl::access::mode::read>(cgh));
    cgh.set_arg(1, output_buffer.get_access<sycl::access::mode::write>(cgh));
    cgh.parallel_for(sycl::range<1>{N}, kernel);
  });

  sycl::host_accessor out{output_buffer};
  for (int i = 0; i < N; i++) {
    assert(out[i] == ((i * multiplier) + added));
  }
}

template <typename T>
void testParam(sycl::queue &q, const sycl::kernel &kernel) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  assert(num_args == 4 && "kernel should take 4 args");

  // Kernel computes sum of squared inputs.
  const T a = 2;
  const T b = 5;
  const T expected = (a * a) + (b * b);

  sycl::buffer a_buffer(&a, sycl::range<1>(1));

  T *const b_ptr = sycl::malloc_shared<T>(1, q);
  b_ptr[0] = b;

  T output{};
  sycl::buffer output_buffer(&output, sycl::range<1>(1));

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> local(1, cgh);
    // Pass T for scalar parameter.
    cgh.set_arg(0, a);
    // Pass USM pointer for OpTypePointer(CrossWorkgroup) parameter.
    cgh.set_arg(1, b_ptr);
    // Pass sycl::accessor for OpTypePointer(CrossWorkgroup) parameter.
    cgh.set_arg(
        2, output_buffer.template get_access<sycl::access::mode::write>(cgh));
    // Pass sycl::local_accessor for OpTypePointer(Workgroup) parameter.
    cgh.set_arg(3, local);
    cgh.parallel_for(sycl::range<1>{1}, kernel);
  });

  sycl::host_accessor out{output_buffer};
  assert(out[0] == expected);
  sycl::free(b_ptr, q);
}

void testKernelsFromSpvFile(std::string file_name) {
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

  const auto getKernel = [&](const std::string &name) {
    return kb_exe.ext_oneapi_get_kernel(name);
  };

  // Test simple kernel
  testSimpleKernel(q, getKernel("my_kernel"), 2, 100);

  // Test OpTypeIntN parameters
  testParam<std::int8_t>(q, getKernel("OpTypeInt8"));
  testParam<std::int16_t>(q, getKernel("OpTypeInt16"));
  testParam<std::int32_t>(q, getKernel("OpTypeInt32"));
  testParam<std::int64_t>(q, getKernel("OpTypeInt64"));

  // Test OpTypeFloatN parameters
  if (q.get_device().has(sycl::aspect::fp16)) {
    testParam<sycl::half>(q, getKernel("OpTypeFloat16"));
  }
  testParam<float>(q, getKernel("OpTypeFloat32"));
  if (q.get_device().has(sycl::aspect::fp64)) {
    testParam<double>(q, getKernel("OpTypeFloat64"));
  }
}

int main(int argc, char **argv) {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_SPIRV
  static_assert(false, "KernelCompiler SPIR-V feature test macro undefined");
#endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  assert(argc == 2 && "Usage: ./%t.out <kernels-spv-file>");
  testKernelsFromSpvFile(argv[1]);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
