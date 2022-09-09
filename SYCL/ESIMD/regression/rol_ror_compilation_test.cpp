// REQUIRES: gpu-intel-gen11
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: esimd_emulator
//==- rol_ror_compilation_test.cpp - Test for compilation of rol/ror functions
// functions -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>
using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

#define SIMD 16
#define THREAD_NUM 512
#define TEST_VALUE 0x55555555

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <typename TRes, typename TArg1, typename TArg2,
          class Sat = __ESIMD_NS::saturation_off_tag>
int test(cl::sycl::queue q, TArg1 arg1, TArg2 arg2, TRes expected_rol,
         TRes expected_ror) {
  shared_allocator<TArg1> allocator1(q);
  shared_vector<TArg1> input(THREAD_NUM * SIMD, allocator1);

  for (int i = 0; i < input.size(); ++i) {
    input[i] = arg1;
  }

  shared_allocator<TRes> allocator0(q);
  shared_vector<TRes> rol_output(THREAD_NUM * SIMD, allocator0);
  shared_vector<TRes> ror_output(THREAD_NUM * SIMD, allocator0);

  shared_vector<TRes> scalar_rol_output(1, allocator0);
  shared_vector<TRes> scalar_ror_output(1, allocator0);

  nd_range<1> Range((range<1>(THREAD_NUM)), (range<1>(SIMD)));
  auto e = q.submit([&](handler &cgh) {
    TArg1 *in_ptr = input.data();

    TRes *rol_ptr = rol_output.data();
    TRes *ror_ptr = ror_output.data();

    TRes *scalar_rol_ptr = scalar_rol_output.data();
    TRes *scalar_ror_ptr = scalar_ror_output.data();

    cgh.parallel_for(Range, [=](nd_item<1> it) SYCL_ESIMD_FUNCTION {
      Sat sat;

      __ESIMD_NS::simd<TArg1, SIMD> input_vec;
      input_vec.copy_from(in_ptr + it.get_global_id(0) * SIMD);

      __ESIMD_NS::simd<TRes, 1> scalar_result;
      __ESIMD_NS::simd<TRes, SIMD> result;

      result = rol<TArg1>(input_vec, arg2);
      scalar_result = rol<TArg1>(arg1, arg2);
      result.copy_to(rol_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_rol_ptr);

      result = ror<TArg1>(input_vec, arg2);
      scalar_result = ror<TArg1>(arg1, arg2);
      result.copy_to(ror_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_ror_ptr);
    });
  });
  e.wait();

  for (int i = 0; i < THREAD_NUM * SIMD; ++i) {
    if (rol_output[i] ^ expected_rol || scalar_rol_output[0] ^ expected_rol) {
      std::cout << "Test failed for rol test " << input[i] << "." << std::endl;
      return 1;
    }
    if (ror_output[i] ^ expected_ror || scalar_ror_output[0] ^ expected_ror) {
      std::cout << "Test failed for ror test " << input[i] << "." << std::endl;
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(properties);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  int test_result = 0;

  test_result |= test<uint32_t, uint32_t, int32_t>(q, TEST_VALUE, 1, 0xAAAAAAAA,
                                                   0xAAAAAAAA);
  test_result |= test<uint32_t, int32_t, int32_t>(q, TEST_VALUE, 1, 0xAAAAAAAA,
                                                  0xAAAAAAAA);
  test_result |= test<int32_t, uint32_t, int32_t>(q, TEST_VALUE, 1, 0xAAAAAAAA,
                                                  0xAAAAAAAA);
  test_result |=
      test<int32_t, int32_t, int32_t>(q, TEST_VALUE, 1, 0xAAAAAAAA, 0xAAAAAAAA);

  test_result |=
      test<uint32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0xAAAAAAAA, 0xAAAAAAAA);
  test_result |=
      test<uint32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0xAAAAAAAA, 0xAAAAAAAA);
  test_result |=
      test<int32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0xAAAAAAAA, 0xAAAAAAAA);
  test_result |= test<int32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
      q, TEST_VALUE, 1, 0xAAAAAAAA, 0xAAAAAAAA);

  test_result |= test<uint16_t, uint16_t, int32_t>(q, 0x8000, 1, 0x1, 0x4000);
  test_result |= test<uint32_t, int16_t, int32_t>(q, 0x8000, 1, 0x1, 0x4000);

  test_result |= test<uint16_t, uint16_t, int32_t>(q, 0x4000, 2, 0x1, 0x1000);
  test_result |=
      test<uint16_t, uint16_t, int32_t>(q, 0x7FFF, 2, 0xFFFD, 0xDFFF);

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
