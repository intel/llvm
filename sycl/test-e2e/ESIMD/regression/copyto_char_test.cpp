// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: opencl && gpu-intel-pvc

//==- copyto_char_test.cpp - Test for using copy_to to copy char buffers -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

#include <iostream>
#include <memory>

using namespace sycl::ext::intel::esimd;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

using DataT = char;

template <int NumElems, bool IsAcc, int ResultOffset = 0> int test_to_copy() {
  sycl::queue queue;
  constexpr int NumSelectedElems = NumElems / 3;
  constexpr int Stride = 2;
  constexpr int Offset = 4;

  shared_allocator<DataT> allocator(queue);
  shared_vector<DataT> result(NumElems + ResultOffset, allocator);
  shared_vector<DataT> initial_ref_data(NumElems, allocator);
  shared_vector<DataT> ref_data_for_fill(NumSelectedElems, allocator);

  constexpr size_t value_for_increase_ref_data_for_fill = 50;
  for (size_t i = 0; i < NumElems; ++i) {
    initial_ref_data[i] = i + 1;
  }
  // We should have different values in the first reference data and in the
  // second reference data.
  for (size_t i = 0; i < NumSelectedElems; ++i) {
    ref_data_for_fill[i] =
        initial_ref_data[i] + value_for_increase_ref_data_for_fill;
  }
  if constexpr (IsAcc) {
    sycl::buffer<DataT> output_buf(result.data() + ResultOffset,
                                   result.size() - ResultOffset);
    queue.submit([&](sycl::handler &cgh) {
      DataT *init_ref_ptr = initial_ref_data.data();
      DataT *ref_data_for_fill_ptr = ref_data_for_fill.data();
      auto acc =
          output_buf.template get_access<sycl::access::mode::read_write>(cgh);

      cgh.single_task([=]() SYCL_ESIMD_KERNEL {
        simd<DataT, NumElems> src_simd_obj;
        src_simd_obj.copy_from(init_ref_ptr);
        simd<DataT, NumSelectedElems> dst_simd_obj;
        dst_simd_obj.copy_from(ref_data_for_fill_ptr);
        src_simd_obj.template select<NumSelectedElems, Stride>(Offset) =
            dst_simd_obj;
        src_simd_obj.copy_to(acc, 0);
      });
    });
  } else {
    queue.submit([&](sycl::handler &cgh) {
      DataT *init_ref_ptr = initial_ref_data.data();
      DataT *ref_data_for_fill_ptr = ref_data_for_fill.data();
      DataT *const out_ptr = result.data() + ResultOffset;

      cgh.single_task([=]() SYCL_ESIMD_KERNEL {
        simd<DataT, NumElems> src_simd_obj;
        src_simd_obj.copy_from(init_ref_ptr);
        simd<DataT, NumSelectedElems> dst_simd_obj;
        dst_simd_obj.copy_from(ref_data_for_fill_ptr);
        src_simd_obj.template select<NumSelectedElems, Stride>(Offset) =
            dst_simd_obj;
        src_simd_obj.copy_to(out_ptr);
      });
    });
  }
  queue.wait_and_throw();

  std::vector<size_t> selected_indices;
  // Collect the indexess that has been selected.
  for (size_t i = Offset; i < Stride * NumSelectedElems + Offset; i += Stride) {
    selected_indices.push_back(i);
  }

  // Push the largest value to avoid the following error: can't dereference
  // out of range vector iterator.
  selected_indices.push_back(std::numeric_limits<size_t>::max());
  auto selected_indices_ptr = selected_indices.begin();

  // Verify that values, that do not was selected has initial values.
  for (size_t i = 0; i < NumElems; ++i) {
    // If current index is less than selected index verify that this element
    // hasn't been selected and changed.
    if (i < *selected_indices_ptr) {
      DataT expected = initial_ref_data[i];

      DataT retrieved = result[i + ResultOffset];

      if (expected != retrieved) {
        std::cout << "Test failed, retrieved value: "
                  << static_cast<int>(retrieved)
                  << ", but expected: " << static_cast<int>(expected)
                  << ", at index: " << i << std::endl;
        return 1;
      }
    } else {

      DataT expected = ref_data_for_fill[(i - Offset) / Stride];

      DataT retrieved = result[i + ResultOffset];

      if (expected != retrieved) {
        std::cout << "Test failed, retrieved value: "
                  << static_cast<int>(retrieved)
                  << ", but expected: " << static_cast<int>(expected)
                  << ", at index: " << i << std::endl;
        return 1;
      }
      selected_indices_ptr++;
    }
  }

  return 0;
}

int main() {
  int test_result = 0;

  test_result |= test_to_copy<16, false>();
  test_result |= test_to_copy<32 + 8, false>();
  test_result |= test_to_copy<32 + 10, false>();
  test_result |= test_to_copy<10, false>();
  test_result |= test_to_copy<8, false>();
  test_result |= test_to_copy<7, false>();
  test_result |= test_to_copy<15, false>();
  test_result |= test_to_copy<32, false>();

  test_result |= test_to_copy<16, true>();
  test_result |= test_to_copy<32 + 8, true>();
  test_result |= test_to_copy<32 + 10, true>();
  test_result |= test_to_copy<10, true>();
  test_result |= test_to_copy<8, true>();
  test_result |= test_to_copy<7, true>();
  test_result |= test_to_copy<15, true>();
  test_result |= test_to_copy<32, true>();

  test_result |= test_to_copy<16, false, 1>();
  test_result |= test_to_copy<32 + 8, false, 1>();
  test_result |= test_to_copy<32 + 10, false, 1>();
  test_result |= test_to_copy<10, false, 1>();
  test_result |= test_to_copy<8, false, 1>();
  test_result |= test_to_copy<7, false, 1>();
  test_result |= test_to_copy<15, false, 1>();
  test_result |= test_to_copy<32, false, 1>();

  test_result |= test_to_copy<16, true, 1>();
  test_result |= test_to_copy<32 + 8, true, 1>();
  test_result |= test_to_copy<32 + 10, true, 1>();
  test_result |= test_to_copy<10, true, 1>();
  test_result |= test_to_copy<8, true, 1>();
  test_result |= test_to_copy<7, true, 1>();
  test_result |= test_to_copy<15, true, 1>();
  test_result |= test_to_copy<32, true, 1>();

  if (!test_result) {
    std::cout << "Test passed" << std::endl;
  }

  return test_result;
}
