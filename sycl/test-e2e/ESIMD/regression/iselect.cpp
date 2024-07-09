//==--- iselect.cpp - DPC++ ESIMD feature test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is basic test for testing proper intrinsics generation.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

#include <iostream>
#include <typeinfo>

namespace esimd = sycl::ext::intel::esimd;
template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

int test(sycl::queue queue) {

  shared_allocator<int32_t> allocator1(queue);
  shared_vector<int32_t> input(16, allocator1);
  shared_vector<int32_t> output(16, allocator1);
  shared_allocator<uint16_t> allocator2(queue);
  shared_vector<uint16_t> offsets(8, allocator2);

  for (int i = 0; i < input.size(); ++i) {
    input[i] = i;
  }

  offsets[0] = 3;
  offsets[1] = 5;
  offsets[2] = 9;
  offsets[3] = 2;
  offsets[4] = 7;
  offsets[5] = 8;
  offsets[6] = 0;
  offsets[7] = 1;
  {
    queue
        .submit([&](sycl::handler &cgh) {
          int32_t *in_ptr = input.data();
          int32_t *out_ptr = output.data();
          uint16_t *off_ptr = offsets.data();
          cgh.single_task([=]() SYCL_ESIMD_KERNEL {
            __ESIMD_NS::simd<int32_t, 16> input_load_vec;
            input_load_vec.copy_from(in_ptr);

            esimd::simd<uint16_t, 8> offset;
            offset.copy_from(off_ptr);

            esimd::simd<int, 8> data = input_load_vec.iselect(offset);
            data += 3;
            input_load_vec.iupdate(offset, data,
                                            esimd::simd_mask<8>(1));
            input_load_vec.copy_to(out_ptr);
          });
        })
        .wait_and_throw();
  }
  int32_t expected_values[] = {3,  4,  5,  6,  4,  8,  6,  10,
                               11, 12, 10, 11, 12, 13, 14, 15};
  int error = 0;
  for (int i = 0; i < 16; i++) {
    if (output[i] != expected_values[i]) {
      ++error;
      std::cout << "Output[" << i << "] = " << output[i] << ", expected "
                << expected_values[i] << std::endl;
    }
  }

  return error;
}

int main(int, char **) {
  sycl::queue queue;

  auto dev = queue.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  int test_result = 0;

  test_result = test(queue);

  if (test_result == 0) {
    std::cout << "Pass!!" << std::endl;
  } else {
    std::cout << "Fail!!" << std::endl;
  }

  return test_result;
}
