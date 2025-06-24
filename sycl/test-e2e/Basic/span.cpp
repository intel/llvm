// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: aspect-usm_shared_allocations, aspect-usm_device_allocations,
// aspect-usm_host_allocations

//==--------------- span.cpp - SYCL span E2E device tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests SYCL-specific span functionality requiring device execution:
// - USM memory allocation and span creation
// - Device lambda capture of spans
// - Kernel execution with span parameters
// - Read/write operations on device
//

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

#include <sycl/detail/core.hpp>

#include <sycl/sycl_span.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

using namespace sycl;

namespace BasicUSMTests {

void testSpanFromUSMSharedAllocation() {
  queue q;

  constexpr size_t num_elements = 5;
  auto *usm_data = malloc_shared<int>(num_elements, q);
  std::iota(usm_data, usm_data + num_elements, 1);

  q.submit([&](handler &cgh) {
     cgh.single_task<class usm_span_test>([=] {
       sycl::span<int> device_span(usm_data, num_elements);

       assert(device_span.size() == num_elements);
       assert(!device_span.empty());

       for (size_t i = 0; i < device_span.size(); ++i) {
         assert(device_span[i] == static_cast<int>(i + 1));
       }

       assert(device_span.front() == 1);
       assert(device_span.back() == 5);
     });
   }).wait();

  free(usm_data, q);
}

} // namespace BasicUSMTests

namespace DeviceCaptureTests {

void testSpanCaptureAndModification() {
  queue q;

  constexpr size_t num_read_tests = 2;
  buffer<int, 1> read_results_buffer((range<1>(num_read_tests)));

  using usm_vec_allocator = usm_allocator<int, usm::alloc::shared>;
  usm_vec_allocator vec_allocator(q);
  std::vector<int, usm_vec_allocator> usm_vector(4, vec_allocator);
  std::iota(usm_vector.begin(), usm_vector.end(), 1);
  sycl::span<int> vector_span{usm_vector};

  constexpr int first_modification = 100;
  vector_span[0] += first_modification;

  auto *raw_usm_data = malloc_shared<int>(4, q);
  sycl::span<int> raw_usm_span(raw_usm_data, 4);
  std::iota(raw_usm_span.begin(), raw_usm_span.end(), 1);
  raw_usm_span[0] += first_modification;

  constexpr int second_modification = 1000;

  event kernel_event = q.submit([&](handler &cgh) {
    auto results_acc = read_results_buffer.get_access<access::mode::write>(cgh);

    cgh.single_task<class span_capture_test>([=] {
      results_acc[0] = vector_span[0];
      results_acc[1] = raw_usm_span[0];

      vector_span[1] += second_modification;
      raw_usm_span[1] += second_modification;
    });
  });
  kernel_event.wait();

  {
    host_accessor read_results(read_results_buffer, read_only);
    for (int i = 0; i < num_read_tests; i++) {
      assert(read_results[i] == (1 + first_modification) &&
             "Read operation should have returned 101");
    }
  }

  assert(vector_span[1] == (2 + second_modification) &&
         "Vector span write should have resulted in 1002");
  assert(raw_usm_span[1] == (2 + second_modification) &&
         "Raw USM span write should have resulted in 1002");

  free(raw_usm_data, q);
}

void fillSpanWithValue(sycl::span<int> target_span, int fill_value) {
  for (auto &element : target_span) {
    element = fill_value;
  }
}

void testSpanCreationOnDevice() {
  queue q;

  constexpr size_t array_size = 4;
  buffer<int, 1> verification_buffer((range<1>(array_size)));

  event kernel_event = q.submit([&](handler &cgh) {
    auto verification_acc =
        verification_buffer.get_access<access::mode::write>(cgh);

    cgh.single_task<class device_span_creation>([=] {
      int device_array[array_size] = {1, 2, 3, 4};
      sycl::span<int> device_span{device_array};

      constexpr int fill_value = 10;
      fillSpanWithValue(device_span, fill_value);

      for (size_t i = 0; i < array_size; i++) {
        verification_acc[i] = device_span[i];
      }
    });
  });
  kernel_event.wait();

  {
    host_accessor verification_acc(verification_buffer, read_only);
    for (size_t i = 0; i < array_size; i++) {
      assert(verification_acc[i] == 10 &&
             "All elements should have been set to 10");
    }
  }
}

} // namespace DeviceCaptureTests

namespace BufferSpanTests {

void testSpanFromBufferAccessor() {
  constexpr size_t num_elements = 1024;
  std::vector<float> host_data(num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i) * 0.5f;
  }

  buffer<float, 1> data_buffer(host_data.data(), range<1>(num_elements));

  queue q;

  q.submit([&](handler &cgh) {
     auto data_acc = data_buffer.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(range<1>(num_elements), [=](id<1> idx) {
       auto *raw_ptr =
           data_acc.template get_multi_ptr<access::decorated::no>().get();
       span<float> accessor_span(raw_ptr, num_elements);

       accessor_span[idx] *= 2.0f;
     });
   }).wait();

  auto process_span_section = [](span<float> data_span, size_t start_idx,
                                 size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      data_span[i] = std::sqrt(data_span[i]);
    }
  };

  constexpr size_t num_chunks = 4;
  q.submit([&](handler &cgh) {
     auto data_acc = data_buffer.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(range<1>(num_chunks), [=](id<1> chunk_id) {
       auto *raw_ptr =
           data_acc.template get_multi_ptr<access::decorated::no>().get();
       span<float> full_span(raw_ptr, num_elements);

       size_t chunk_size = num_elements / num_chunks;
       size_t start_idx = chunk_id[0] * chunk_size;
       size_t end_idx = (chunk_id[0] == num_chunks - 1)
                            ? num_elements
                            : start_idx + chunk_size;

       process_span_section(full_span, start_idx, end_idx);
     });
   }).wait();

  float computed_sum = 0.0f;
  {
    auto host_acc = data_buffer.get_host_access(read_only);
    span<const float> const_span(host_acc.get_pointer(), num_elements);

    for (const auto &value : const_span) {
      computed_sum += value;
    }
  }

  float expected_sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    expected_sum += std::sqrt(static_cast<float>(i));
  }

  assert(std::abs(computed_sum - expected_sum) < 0.01f &&
         "Computed sum doesn't match expected value");
}

void testSpanWith2DBuffer() {
  constexpr size_t height = 32;
  constexpr size_t width = 64;

  buffer<int, 2> buffer_2d(range<2>(height, width));

  queue q;

  q.submit([&](handler &cgh) {
     auto buf_acc = buffer_2d.get_access<access::mode::write>(cgh);

     cgh.parallel_for(range<2>(height, width), [=](id<2> idx) {
       buf_acc[idx] = idx[0] * width + idx[1];
     });
   }).wait();

  q.submit([&](handler &cgh) {
     auto buf_acc = buffer_2d.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(range<1>(height), [=](id<1> row_idx) {
       auto *row_start =
           buf_acc.template get_multi_ptr<access::decorated::no>().get() +
           row_idx[0] * width;
       span<int> row_span(row_start, width);

       for (size_t i = 0; i < width / 2; ++i) {
         std::swap(row_span[i], row_span[width - 1 - i]);
       }
     });
   }).wait();

  {
    auto host_acc = buffer_2d.get_host_access(read_only);
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        int expected_value = row * width + (width - 1 - col);
        assert(host_acc[row][col] == expected_value &&
               "Row reversal verification failed");
      }
    }
  }
}

void testSubspanFromAccessor() {
  constexpr size_t buffer_size = 100;
  buffer<int, 1> data_buffer((range<1>(buffer_size)));

  queue q;

  q.submit([&](handler &cgh) {
     auto buf_acc = data_buffer.get_access<access::mode::write>(cgh);

     cgh.single_task<class init_buffer>([=] {
       span<int> full_span(
           buf_acc.template get_multi_ptr<access::decorated::no>().get(),
           buffer_size);

       for (size_t i = 0; i < full_span.size(); ++i) {
         full_span[i] = static_cast<int>(i);
       }

       auto first_half = full_span.subspan(0, buffer_size / 2);
       auto second_half = full_span.subspan(buffer_size / 2);

       for (auto &elem : first_half) {
         elem += 100;
       }

       for (auto &elem : second_half) {
         elem += 200;
       }
     });
   }).wait();

  {
    auto host_acc = data_buffer.get_host_access(read_only);
    for (size_t i = 0; i < buffer_size; ++i) {
      int expected = (i < buffer_size / 2) ? i + 100 : i + 200;
      assert(host_acc[i] == expected && "Subspan modification failed");
    }
  }
}

} // namespace BufferSpanTests

namespace USMMemoryTypeTests {

void testSpanWithAllUSMTypes() {
  queue q;
  constexpr size_t data_size = 256;

  // Test 1: Device Memory Span
  int *device_memory = malloc_device<int>(data_size, q);
  span<int> device_mem_span(device_memory, data_size);

  q.parallel_for(range<1>(data_size), [=](id<1> idx) {
     device_mem_span[idx] = idx[0];
   }).wait();

  // Test 2: Host Memory Span
  int *host_memory = malloc_host<int>(data_size, q);
  span<int> host_mem_span(host_memory, data_size);

  for (size_t i = 0; i < data_size; ++i) {
    host_mem_span[i] = i * 2;
  }

  // Test 3: Shared Memory Span
  int *shared_memory = malloc_shared<int>(data_size, q);
  span<int> shared_mem_span(shared_memory, data_size);

  q.parallel_for(range<1>(data_size), [=](id<1> idx) {
     shared_mem_span[idx] = host_mem_span[idx];
   }).wait();

  bool copy_verified = true;
  for (size_t i = 0; i < data_size; ++i) {
    if (shared_mem_span[i] != static_cast<int>(i * 2)) {
      copy_verified = false;
      break;
    }
  }
  assert(copy_verified && "Host to shared memory copy via span failed");

  // Test 4: Local Memory Span in Work-Groups
  constexpr size_t work_group_size = 64;
  constexpr size_t num_work_groups = data_size / work_group_size;

  q.submit([&](handler &cgh) {
     local_accessor<int, 1> local_mem_acc(range<1>(work_group_size), cgh);

     cgh.parallel_for(
         nd_range<1>(data_size, work_group_size), [=](nd_item<1> item) {
           auto local_id = item.get_local_id(0);
           auto group_id = item.get_group(0);

           auto *local_ptr =
               local_mem_acc.get_multi_ptr<access::decorated::no>().get();
           span<int> local_mem_span(local_ptr, work_group_size);

           local_mem_span[local_id] = local_id;

           item.barrier(access::fence_space::local_space);

           if (local_id == 0) {
             int local_sum = 0;
             for (size_t i = 0; i < work_group_size; ++i) {
               local_sum += local_mem_span[i];
             }
             shared_mem_span[group_id] = local_sum;
           }
         });
   }).wait();

  constexpr int expected_local_sum =
      (work_group_size - 1) * work_group_size / 2;
  for (size_t i = 0; i < num_work_groups; ++i) {
    assert(shared_mem_span[i] == expected_local_sum &&
           "Local memory span sum computation incorrect");
  }

  // Test 5: Subspan Operations with USM Memory
  auto subspan_test = shared_mem_span.subspan(10, 20);

  assert(subspan_test.size() == 20 && "Subspan size should be 20");
  assert(subspan_test.data() == shared_mem_span.data() + 10 &&
         "Subspan should point to offset +10");

  auto first_10 = shared_mem_span.first(10);
  auto last_10 = shared_mem_span.last(10);

  assert(first_10.size() == 10 && "first(10) should return span of size 10");
  assert(last_10.size() == 10 && "last(10) should return span of size 10");
  assert(first_10.data() == shared_mem_span.data() &&
         "first() should point to beginning");
  assert(last_10.data() == shared_mem_span.data() + data_size - 10 &&
         "last() should point to end - 10");

  free(device_memory, q);
  free(host_memory, q);
  free(shared_memory, q);
}

void testMixedMemoryTypeOperations() {
  queue q;
  constexpr size_t test_size = 128;

  float *device_mem = malloc_device<float>(test_size, q);
  float *shared_mem = malloc_shared<float>(test_size, q);

  span<float> device_span(device_mem, test_size);
  span<float> shared_span(shared_mem, test_size);

  for (size_t i = 0; i < test_size; ++i) {
    shared_span[i] = static_cast<float>(i) * 0.1f;
  }

  q.parallel_for(range<1>(test_size), [=](id<1> idx) {
     device_span[idx] = shared_span[idx] * 2.0f;
   }).wait();

  q.parallel_for(range<1>(test_size), [=](id<1> idx) {
     shared_span[idx] = device_span[idx] * device_span[idx];
   }).wait();

  bool test_passed = true;
  for (size_t i = 0; i < test_size; ++i) {
    float expected = (i * 0.1f * 2.0f) * (i * 0.1f * 2.0f);
    if (std::abs(shared_span[i] - expected) > 0.001f) {
      test_passed = false;
      break;
    }
  }
  assert(test_passed && "Mixed memory type operation failed");

  free(device_mem, q);
  free(shared_mem, q);
}

} // namespace USMMemoryTypeTests

int main() {
  BasicUSMTests::testSpanFromUSMSharedAllocation();

  DeviceCaptureTests::testSpanCaptureAndModification();
  DeviceCaptureTests::testSpanCreationOnDevice();

  BufferSpanTests::testSpanFromBufferAccessor();
  BufferSpanTests::testSpanWith2DBuffer();
  BufferSpanTests::testSubspanFromAccessor();

  USMMemoryTypeTests::testSpanWithAllUSMTypes();
  USMMemoryTypeTests::testMixedMemoryTypeOperations();

  return 0;
}
