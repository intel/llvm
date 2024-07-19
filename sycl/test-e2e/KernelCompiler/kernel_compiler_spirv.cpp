//==- kernel_compiler_spirv.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/Kernels/kernels.spv %S/Kernels/kernels_fp16.spv %S/Kernels/kernels_fp64.spv

// Test case for the sycl_ext_oneapi_kernel_compiler_spirv extension. This test
// loads pre-compiled kernels from a SPIR-V file and runs them.

#include <array>
#include <cassert>
#include <fstream>
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

sycl::kernel_bundle<sycl::bundle_state::executable>
loadKernelsFromFile(sycl::queue &q, std::string file_name) {
  namespace syclex = sycl::ext::oneapi::experimental;

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
  return kb_exe;
}

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

void testStruct(sycl::queue &q, const sycl::kernel &kernel) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  assert(num_args == 2 && "kernel should take 2 args");

  // This definition must match the one used in the kernel.
  struct S {
    std::int32_t i;
    cl_float f;
    std::int32_t *p;
    struct Inner {
      std::int32_t i;
      float f;
      std::int32_t *p;
    } inner;
  };

  // Any constants can be used to initialize this input.
  std::int32_t *const in_p0 = sycl::malloc_shared<std::int32_t>(1, q);
  std::int32_t *const in_p1 = sycl::malloc_shared<std::int32_t>(1, q);
  *in_p0 = 3;
  *in_p1 = 6;
  S input{1, 2.0f, in_p0, S::Inner{4, 5.0f, in_p1}};

  std::int32_t *const out_p0 = sycl::malloc_shared<std::int32_t>(1, q);
  std::int32_t *const out_p1 = sycl::malloc_shared<std::int32_t>(1, q);
  *out_p0 = 0;
  *out_p1 = 0;
  S *output = sycl::malloc_shared<S>(1, q);
  *output = S{0, 0, out_p0, S::Inner{0, 0, out_p1}};

  q.submit([&](sycl::handler &cgh) {
     cgh.set_arg(0, input);
     cgh.set_arg(1, output);
     cgh.parallel_for(sycl::range<1>{1}, kernel);
   }).wait();

  std::cout << "output->i: " << output->i << std::endl;
  std::cout << "output->f: " << output->f << std::endl;
  std::cout << "output->p: " << *(output->p) << std::endl;
  std::cout << "output->inner.i: " << output->inner.i << std::endl;
  std::cout << "output->inner.f: " << output->inner.f << std::endl;
  std::cout << "output->inner.p: " << *(output->inner.p) << std::endl;

  // For each scalar struct member, output == (2 * input). For pointer members,
  // *output == (2 * (*input)).
  assert(output->i == input.i * 2);
  assert(output->f == input.f * 2);
  assert(*output->p == (*input.p) * 2);
  assert(output->inner.i == input.inner.i * 2);
  assert(output->inner.f == input.inner.f * 2);
  assert(*output->inner.p == (*input.inner.p) * 2);

  sycl::free(output, q);
  sycl::free(in_p0, q);
  sycl::free(in_p1, q);
  sycl::free(out_p0, q);
  sycl::free(out_p1, q);
}

void testKernelsFromSpvFile(std::string kernels_file,
                            std::string fp16_kernel_file,
                            std::string fp64_kernel_file) {
  const auto getKernel =
      [](sycl::kernel_bundle<sycl::bundle_state::executable> &bundle,
         const std::string &name) {
        return bundle.ext_oneapi_get_kernel(name);
      };

  sycl::queue q;
  auto bundle = loadKernelsFromFile(q, kernels_file);

  // Test simple kernel.
  testSimpleKernel(q, getKernel(bundle, "my_kernel"), 2, 100);

  // Test parameters.
  testParam<std::int8_t>(q, getKernel(bundle, "OpTypeInt8"));
  testParam<std::int16_t>(q, getKernel(bundle, "OpTypeInt16"));
  testParam<std::int32_t>(q, getKernel(bundle, "OpTypeInt32"));
  testParam<std::int64_t>(q, getKernel(bundle, "OpTypeInt64"));
  testParam<float>(q, getKernel(bundle, "OpTypeFloat32"));

  // Test OpTypeFloat16 parameters.
  if (q.get_device().has(sycl::aspect::fp16)) {
    auto fp16_bundle = loadKernelsFromFile(q, fp16_kernel_file);
    testParam<sycl::half>(q, getKernel(fp16_bundle, "OpTypeFloat16"));
  }

  // Test OpTypeFloat64 parameters.
  if (q.get_device().has(sycl::aspect::fp64)) {
    auto fp64_bundle = loadKernelsFromFile(q, fp64_kernel_file);
    testParam<double>(q, getKernel(fp64_bundle, "OpTypeFloat64"));
  }

  // Test OpTypeStruct parameters.
  testStruct(q, getKernel(bundle, "OpTypeStruct"));
}

int main(int argc, char **argv) {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_SPIRV
  static_assert(false, "KernelCompiler SPIR-V feature test macro undefined");
#endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  assert(argc == 4 && "Usage: ./%t.out <kernels-spv-file> "
                      "<fp16-kernel-spv-file> <fp64-kernel-spv-file>");
  testKernelsFromSpvFile(argv[1], argv[2], argv[3]);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
