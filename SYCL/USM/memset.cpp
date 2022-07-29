// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==---- memset.cpp - USM memset test --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

using namespace sycl;

static constexpr int N = 100;
static constexpr int VAL = 3;

int main() {
  queue q([](exception_list el) {
    for (auto &e : el)
      std::rethrow_exception(e);
  });
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  char *array;

  if (dev.get_info<info::device::usm_host_allocations>()) {
    // Test memset on host
    array = (char *)malloc_host(N * sizeof(char), q);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();
    for (int i = 0; i < N; ++i) {
      assert(array[i] == VAL);
    }
    free(array, ctxt);

    // Test memset on aligned host
    array =
        (char *)aligned_alloc_host(alignof(long long), N * sizeof(char), ctxt);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();
    for (int i = 0; i < N; ++i) {
      assert(array[i] == VAL);
    }
    free(array, ctxt);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    // Test memset on shared
    array = (char *)malloc_shared(N * sizeof(char), q);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();
    for (int i = 0; i < N; ++i) {
      assert(array[i] == VAL);
    }
    free(array, ctxt);

    // Test memset on aligned shared
    array = (char *)aligned_alloc_shared(alignof(long long), N * sizeof(char),
                                         dev, ctxt);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();
    for (int i = 0; i < N; ++i) {
      assert(array[i] == VAL);
    }
    free(array, ctxt);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    std::vector<char> out;
    out.resize(N);

    // Test memset on device
    array = (char *)malloc_device(N * sizeof(char), q);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();

    {
      buffer<char, 1> buf{&out[0], range<1>{N}};
      q.submit([&](handler &h) {
        auto acc = buf.template get_access<access::mode::write>(h);
        h.parallel_for<class usm_device_transfer>(
            range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
      });
      q.wait_and_throw();
    }

    for (int i = 0; i < N; ++i) {
      assert(out[i] == VAL);
    }
    free(array, ctxt);

    out.clear();
    out.resize(N);

    // Test memset on aligned device
    array = (char *)aligned_alloc_device(alignof(long long), N * sizeof(char),
                                         dev, ctxt);
    q.submit([&](handler &h) {
       h.memset(array, VAL, N * sizeof(char));
     }).wait();

    {
      buffer<char, 1> buf{&out[0], range<1>{N}};
      q.submit([&](handler &h) {
        auto acc = buf.template get_access<access::mode::write>(h);
        h.parallel_for<class usm_aligned_device_transfer>(
            range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
      });
      q.wait_and_throw();
    }

    for (int i = 0; i < N; ++i) {
      assert(out[i] == VAL);
    }
    free(array, ctxt);
  }

  try {
    // Filling to nullptr should throw.
    q.submit([&](handler &cgh) { cgh.memset(nullptr, 0, N * sizeof(char)); });
    q.wait_and_throw();
    assert(false && "Expected error from writing to nullptr");
  } catch (runtime_error e) {
  }

  // Filling to nullptr is skipped if the number of bytes to fill is 0.
  q.submit([&](handler &cgh) { cgh.memset(nullptr, 0, 0); });
  q.wait_and_throw();

  return 0;
}
