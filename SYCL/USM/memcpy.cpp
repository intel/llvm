//==---- memcpy.cpp - USM memcpy test --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include <sycl/sycl.hpp>

using namespace cl::sycl;

static constexpr int N = 100;
static constexpr int VAL = 42;

void init_on_host(queue q, int *arr) {
  for (int i = 0; i < N; ++i) {
    arr[i] = VAL;
  }
}

void check_on_host(queue q, int *arr) {
  for (int i = 0; i < N; ++i) {
    assert(arr[i] == VAL);
  }
}

void init_on_device(queue q, int *arr) {
  q.submit([&](handler &h) {
     h.parallel_for<class usm_device_init>(
         range<1>(N), [=](id<1> item) { arr[item] = VAL; });
   }).wait();
}

void check_on_device(queue q, int *arr) {
  std::vector<int> out;
  out.resize(N);
  {
    buffer<int, 1> buf{&out[0], range<1>{N}};
    q.submit([&](handler &h) {
       auto acc = buf.template get_access<access::mode::write>(h);
       h.parallel_for<class usm_device_transfer>(
           range<1>(N), [=](id<1> item) { acc[item] = arr[item]; });
     }).wait();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == VAL);
  }
}

#define USM_MALLOC(ARR, ALLOC_TYPE)                                            \
  ARR = (int *)malloc_##ALLOC_TYPE(N * sizeof(int), q);

#define USM_ALIGNED_ALLOC_HOST(ARR)                                            \
  ARR = (int *)aligned_alloc_host(alignof(long long), N * sizeof(int), ctxt);

#define USM_ALIGNED_ALLOC_SHARED(ARR)                                          \
  ARR = (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int), dev,  \
                                    ctxt);

#define USM_ALIGNED_ALLOC_DEVICE(ARR)                                          \
  ARR = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int), dev,  \
                                    ctxt);

#define TEST_MEMCPY(FROM_ARR, FROM_INIT, TO_ARR, TO_CHECK)                     \
  {                                                                            \
    FROM_INIT(q, FROM_ARR);                                                    \
    q.submit(                                                                  \
        [&](handler &h) { h.memcpy(TO_ARR, FROM_ARR, N * sizeof(int)); });     \
    q.wait_and_throw();                                                        \
    TO_CHECK(q, TO_ARR);                                                       \
    free(FROM_ARR, ctxt);                                                      \
    free(TO_ARR, ctxt);                                                        \
  }

#define TEST_MEMCPY_TO_NULLPTR(ARR)                                            \
  {                                                                            \
    try {                                                                      \
      /* Copying to nullptr should throw. */                                   \
      q.submit(                                                                \
          [&](handler &cgh) { cgh.memcpy(nullptr, ARR, sizeof(int) * N); });   \
      q.wait_and_throw();                                                      \
      assert(false && "Expected error from copying to nullptr");               \
    } catch (runtime_error e) {                                                \
    }                                                                          \
    /* Copying to nullptr should throw. */                                     \
    q.submit([&](handler &cgh) { cgh.memcpy(nullptr, ARR, 0); });              \
    q.wait_and_throw();                                                        \
    free(ARR, ctxt);                                                           \
  }

int main() {
  queue q([](exception_list el) {
    for (auto &e : el)
      std::rethrow_exception(e);
  });
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  int *inArray;
  int *outArray;

  if (dev.get_info<info::device::usm_host_allocations>()) {
    // Test host to host
    USM_MALLOC(inArray, host)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned host to aligned host
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test host to aligned host
    USM_MALLOC(inArray, host)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned host to host
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test copy to null from host
    USM_MALLOC(inArray, host)
    TEST_MEMCPY_TO_NULLPTR(inArray)

    // Test copy to null from aligned host
    USM_ALIGNED_ALLOC_HOST(inArray)
    TEST_MEMCPY_TO_NULLPTR(inArray)
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    // Test shared to shared
    USM_MALLOC(inArray, shared)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned shared to aligned shared
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test shared to aligned shared
    USM_MALLOC(inArray, shared)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned shared to shared
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test copy to null from shared
    USM_MALLOC(inArray, shared)
    TEST_MEMCPY_TO_NULLPTR(inArray)

    // Test copy to null from aligned shared
    USM_ALIGNED_ALLOC_SHARED(inArray)
    TEST_MEMCPY_TO_NULLPTR(inArray)
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    // Test device to device
    USM_MALLOC(inArray, device)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_device)

    // Test aligned device to aligned device
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_device)

    // Test device to aligned device
    USM_MALLOC(inArray, shared)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_device)

    // Test aligned device to device
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_device)

    // Test copy to null from device
    USM_MALLOC(inArray, device)
    TEST_MEMCPY_TO_NULLPTR(inArray)

    // Test copy to null from aligned device
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    TEST_MEMCPY_TO_NULLPTR(inArray)
  }

  if (dev.get_info<info::device::usm_host_allocations>() &&
      dev.get_info<info::device::usm_shared_allocations>()) {
    // Test host to shared
    USM_MALLOC(inArray, host)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test shared to host
    USM_MALLOC(inArray, shared)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned host to aligned shared
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned shared to aligned host
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test host to aligned shared
    USM_MALLOC(inArray, host)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test shared to aligned host
    USM_MALLOC(inArray, shared)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned shared to host
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)

    // Test aligned host to shared
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_host)
  }

  if (dev.get_info<info::device::usm_host_allocations>() &&
      dev.get_info<info::device::usm_device_allocations>()) {
    // Test host to device
    USM_MALLOC(inArray, host)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test device to host
    USM_MALLOC(inArray, device)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned host to aligned device
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test aligned device to aligned host
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test host to aligned device
    USM_MALLOC(inArray, host)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test device to aligned host
    USM_MALLOC(inArray, device)
    USM_ALIGNED_ALLOC_HOST(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned device to host
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_MALLOC(outArray, host)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned host to device
    USM_ALIGNED_ALLOC_HOST(inArray)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)
  }

  if (dev.get_info<info::device::usm_host_allocations>() &&
      dev.get_info<info::device::usm_device_allocations>()) {
    // Test shared to device
    USM_MALLOC(inArray, shared)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test device to shared
    USM_MALLOC(inArray, device)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned shared to aligned device
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test aligned device to aligned shared
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test shared to aligned device
    USM_MALLOC(inArray, shared)
    USM_ALIGNED_ALLOC_DEVICE(outArray)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)

    // Test device to aligned shared
    USM_MALLOC(inArray, device)
    USM_ALIGNED_ALLOC_SHARED(outArray)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned device to shared
    USM_ALIGNED_ALLOC_DEVICE(inArray)
    USM_MALLOC(outArray, shared)
    TEST_MEMCPY(inArray, init_on_device, outArray, check_on_host)

    // Test aligned shared to device
    USM_ALIGNED_ALLOC_SHARED(inArray)
    USM_MALLOC(outArray, device)
    TEST_MEMCPY(inArray, init_on_host, outArray, check_on_device)
  }

  return 0;
}

#undef TEST_MEMCPY_TO_NULLPTR
#undef TEST_MEMCPY
#undef USM_ALIGNED_ALLOC_HOST
#undef USM_ALIGNED_ALLOC_SHARED
#undef USM_ALIGNED_ALLOC_DEVICE
#undef USM_MALLOC
