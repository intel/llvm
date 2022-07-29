//==---- fill.cpp - USM fill test ------------------------------------------==//
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

using namespace sycl;

template <typename T> class usm_device_transfer;
template <typename T> class usm_aligned_device_transfer;

static constexpr int N = 100;

struct test_struct {
  short a;
  int b;
  long c;
  long long d;
  sycl::half e;
  float f;
  double g;
};

bool operator==(const test_struct &lhs, const test_struct &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.e == rhs.e && lhs.f == rhs.f && lhs.g == rhs.g;
}

template <typename T>
void runHostTests(device dev, context ctxt, queue q, T val) {
  T *array;

  array = (T *)malloc_host(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);

  array = (T *)aligned_alloc_host(alignof(long long), N * sizeof(T), ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);
}

template <typename T>
void runSharedTests(device dev, context ctxt, queue q, T val) {
  T *array;

  array = (T *)malloc_shared(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);

  array =
      (T *)aligned_alloc_shared(alignof(long long), N * sizeof(T), dev, ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();
  for (int i = 0; i < N; ++i) {
    assert(array[i] == val);
  }
  free(array, ctxt);
}

template <typename T>
void runDeviceTests(device dev, context ctxt, queue q, T val) {
  T *array;
  std::vector<T> out;
  out.resize(N);

  array = (T *)malloc_device(N * sizeof(T), q);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();

  {
    buffer<T, 1> buf{&out[0], range<1>{N}};
    q.submit([&](handler &h) {
       auto acc = buf.template get_access<access::mode::write>(h);
       h.parallel_for<usm_device_transfer<T>>(
           range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
     }).wait();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }
  free(array, ctxt);

  out.clear();
  out.resize(N);

  array =
      (T *)aligned_alloc_device(alignof(long long), N * sizeof(T), dev, ctxt);
  q.submit([&](handler &h) { h.fill(array, val, N); }).wait();

  {
    buffer<T, 1> buf{&out[0], range<1>{N}};
    q.submit([&](handler &h) {
       auto acc = buf.template get_access<access::mode::write>(h);
       h.parallel_for<usm_aligned_device_transfer<T>>(
           range<1>(N), [=](id<1> item) { acc[item] = array[item]; });
     }).wait();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }
  free(array, ctxt);
}

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  test_struct test_obj{4, 42, 424, 4242, 4.2f, 4.242, 4.24242};

  if (dev.get_info<info::device::usm_host_allocations>()) {
    runHostTests<short>(dev, ctxt, q, 4);
    runHostTests<int>(dev, ctxt, q, 42);
    runHostTests<long>(dev, ctxt, q, 424);
    runHostTests<long long>(dev, ctxt, q, 4242);
    runHostTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runHostTests<float>(dev, ctxt, q, 4.242f);
    runHostTests<double>(dev, ctxt, q, 4.24242);
    runHostTests<test_struct>(dev, ctxt, q, test_obj);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    runSharedTests<short>(dev, ctxt, q, 4);
    runSharedTests<int>(dev, ctxt, q, 42);
    runSharedTests<long>(dev, ctxt, q, 424);
    runSharedTests<long long>(dev, ctxt, q, 4242);
    runSharedTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runSharedTests<float>(dev, ctxt, q, 4.242f);
    runSharedTests<double>(dev, ctxt, q, 4.24242);
    runSharedTests<test_struct>(dev, ctxt, q, test_obj);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    runDeviceTests<short>(dev, ctxt, q, 4);
    runDeviceTests<int>(dev, ctxt, q, 42);
    runDeviceTests<long>(dev, ctxt, q, 420);
    runDeviceTests<long long>(dev, ctxt, q, 4242);
    runDeviceTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runDeviceTests<float>(dev, ctxt, q, 4.242f);
    runDeviceTests<double>(dev, ctxt, q, 4.24242);
    runDeviceTests<test_struct>(dev, ctxt, q, test_obj);
  }

  return 0;
}
