// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: esimd_emulator
//==- bit_shift_compilation_test.cpp - Test for compilation of bit shift
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
int test(cl::sycl::queue q, TArg1 arg1, TArg2 arg2, TRes expected_shl,
         TRes expected_shr, TRes expected_lsr, TRes expected_asr) {
  shared_allocator<TArg1> allocator1(q);
  shared_vector<TArg1> input(THREAD_NUM * SIMD, allocator1);

  for (int i = 0; i < input.size(); ++i) {
    input[i] = arg1;
  }

  shared_allocator<TRes> allocator0(q);
  shared_vector<TRes> shl_output(THREAD_NUM * SIMD, allocator0);
  shared_vector<TRes> shr_output(THREAD_NUM * SIMD, allocator0);
  shared_vector<TRes> lsr_output(THREAD_NUM * SIMD, allocator0);
  shared_vector<TRes> asr_output(THREAD_NUM * SIMD, allocator0);

  shared_vector<TRes> scalar_shl_output(1, allocator0);
  shared_vector<TRes> scalar_shr_output(1, allocator0);
  shared_vector<TRes> scalar_lsr_output(1, allocator0);
  shared_vector<TRes> scalar_asr_output(1, allocator0);

  nd_range<1> Range((range<1>(THREAD_NUM)), (range<1>(SIMD)));
  auto e = q.submit([&](handler &cgh) {
    TArg1 *in_ptr = input.data();
    TRes *shl_ptr = shl_output.data();
    TRes *shr_ptr = shr_output.data();
    TRes *lsr_ptr = lsr_output.data();
    TRes *asr_ptr = asr_output.data();

    TRes *scalar_shl_ptr = scalar_shl_output.data();
    TRes *scalar_shr_ptr = scalar_shr_output.data();
    TRes *scalar_lsr_ptr = scalar_lsr_output.data();
    TRes *scalar_asr_ptr = scalar_asr_output.data();

    cgh.parallel_for(Range, [=](nd_item<1> it) SYCL_ESIMD_FUNCTION {
      Sat sat;

      __ESIMD_NS::simd<TArg1, SIMD> input_vec;
      input_vec.copy_from(in_ptr + it.get_global_id(0) * SIMD);

      __ESIMD_NS::simd<TRes, 1> scalar_result;
      __ESIMD_NS::simd<TRes, SIMD> result;

      result = shl<TArg1>(input_vec, arg2, sat);
      scalar_result = shl<TArg1>(arg1, arg2, sat);
      result.copy_to(shl_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_shl_ptr);

      result = shr<TArg1>(input_vec, arg2, sat);
      scalar_result = shr<TArg1>(arg1, arg2, sat);
      result.copy_to(shr_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_shr_ptr);

      result = lsr<TArg1>(input_vec, arg2, sat);
      scalar_result = lsr<TArg1>(arg1, arg2, sat);
      result.copy_to(lsr_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_lsr_ptr);

      result = asr<TArg1>(input_vec, arg2, sat);
      scalar_result = asr<TArg1>(arg1, arg2, sat);
      result.copy_to(asr_ptr + it.get_global_id(0) * SIMD);
      scalar_result.copy_to(scalar_asr_ptr);
    });
  });
  e.wait();

  for (int i = 0; i < THREAD_NUM * SIMD; ++i) {
    if (shl_output[i] ^ expected_shl || scalar_shl_output[0] ^ expected_shl) {
      std::cout << "Test failed for shl test " << input[i] << "." << std::endl;
      return 1;
    }
    if (shr_output[i] ^ expected_shr || scalar_shr_output[0] ^ expected_shr) {
      std::cout << "Test failed for shr test " << input[i] << "." << std::endl;
      return 1;
    }
    if (lsr_output[i] ^ expected_lsr || scalar_lsr_output[0] ^ expected_lsr) {
      std::cout << "Test failed for lsr test " << input[i] << "." << std::endl;
      return 1;
    }
    if (asr_output[i] ^ expected_asr || scalar_asr_output[0] ^ expected_asr) {
      std::cout << "Test failed for asr test " << input[i] << "." << std::endl;
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

  test_result |= test<uint32_t, uint32_t, int32_t>(
      q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |= test<uint32_t, int32_t, int32_t>(
      q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |= test<int32_t, uint32_t, int32_t>(
      q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |= test<int32_t, int32_t, int32_t>(
      q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);

  test_result |=
      test<uint32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |=
      test<uint32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0x7FFFFFFF, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |=
      test<int32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
          q, TEST_VALUE, 1, 0xAAAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);
  test_result |= test<int32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(
      q, TEST_VALUE, 1, 0x7FFFFFFF, 0x2AAAAAAA, 0x2AAAAAAA, 0x2AAAAAAA);

  test_result |= test<uint16_t, uint16_t, int32_t>(q, 0x8000, 1, 0x0, 0x4000,
                                                   0x4000, 0x4000);
  test_result |= test<uint32_t, int16_t, int32_t>(q, 0x8000, 1, 0x0, 0xFFFFC000,
                                                  0xFFFFC000, 0xFFFFC000);

  test_result |= test<uint16_t, uint16_t, int32_t>(q, 0x4000, 2, 0x0, 0x1000,
                                                   0x1000, 0x1000);
  test_result |= test<uint16_t, uint16_t, int32_t>(q, 0x7FFF, 2, 0xFFFC, 0x1FFF,
                                                   0x1FFF, 0x1FFF);

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
