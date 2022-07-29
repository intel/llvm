// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//==- abs_fix_test.cpp - Test for abs function -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>
using namespace sycl;
using namespace sycl::ext::intel::experimental::esimd;

#define SIMD 16
#define THREAD_NUM 512

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

int test_abs(sycl::queue q, int32_t test_value) {

  shared_allocator<int32_t> allocator1(q);
  shared_vector<int32_t> input(THREAD_NUM * SIMD, allocator1);

  for (int i = 0; i < input.size(); ++i) {
    input[i] = test_value;
  }

  shared_allocator<uint32_t> allocator0(q);
  shared_vector<uint32_t> output(THREAD_NUM * SIMD, allocator0);
  shared_vector<uint32_t> scalar_output(1, allocator0);

  nd_range<1> Range((range<1>(THREAD_NUM)), (range<1>(SIMD)));
  auto e = q.submit([&](handler &cgh) {
    int32_t *in_ptr = input.data();
    uint32_t *out_ptr = output.data();
    uint32_t *scalar_ptr = scalar_output.data();
    cgh.parallel_for(Range, [=](nd_item<1> it) SYCL_ESIMD_FUNCTION {
      __ESIMD_NS::simd<int32_t, SIMD> input_load_vec;
      input_load_vec.copy_from(in_ptr + it.get_global_id(0) * SIMD);

      __ESIMD_NS::simd<uint32_t, SIMD> result;
      __ESIMD_NS::simd<uint32_t, 1> scalar_result;
      result = __ESIMD_NS::abs<uint32_t, int32_t, SIMD>(input_load_vec);
      scalar_result = __ESIMD_NS::abs<uint32_t, int32_t, 1>(test_value);
      result.copy_to(out_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_ptr);
    });
  });
  e.wait();

  if (scalar_output[0] != std::abs(test_value)) {
    std::cout << "Test failed for scalar " << test_value << "." << std::endl;
    return 1;
  }
  for (int i = 0; i < THREAD_NUM * SIMD; ++i) {
    if (output[i] != std::abs(input[i])) {
      std::cout << "Test failed for " << input[i] << "." << std::endl;
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(properties);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int test_result = 0;
  test_result |= test_abs(q, 0xFFFFFFFF);

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
