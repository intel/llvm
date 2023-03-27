//==---- fill.cpp - USM fill test ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T> class usm_device_transfer;
template <typename T> class usm_aligned_device_transfer;

static constexpr int N = 100;

struct test_struct_minimum {
  short a;
  int b;
  long c;
  long long d;
  float f;
};

bool operator==(const test_struct_minimum &lhs,
                const test_struct_minimum &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.f == rhs.f;
}

struct test_struct_all : public test_struct_minimum {
  sycl::half e;
  double g;
};

bool operator==(const test_struct_all &lhs, const test_struct_all &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.e == rhs.e && lhs.f == rhs.f && lhs.g == rhs.g;
}

struct test_struct_half : public test_struct_minimum {
  sycl::half e;
};

bool operator==(const test_struct_half &lhs, const test_struct_half &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.e == rhs.e && lhs.f == rhs.f;
}

struct test_struct_double : public test_struct_minimum {
  double g;
};

bool operator==(const test_struct_double &lhs, const test_struct_double &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d &&
         lhs.f == rhs.f && lhs.g == rhs.g;
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

  const bool DoublesSupported = dev.has(sycl::aspect::fp64);
  const bool HalfsSupported = dev.has(sycl::aspect::fp16);

  test_struct_all test_obj_all{4, 42, 424, 4242, 4.2f, 4.242, 4.24242};
  test_struct_half test_obj_half{4, 42, 424, 4242, 4.2f, 4.242};
  test_struct_double test_obj_double{4, 42, 424, 4242, 4.242, 4.24242};
  test_struct_minimum test_obj_minimum{4, 42, 424, 4242, 4.242};

  if (dev.get_info<info::device::usm_host_allocations>()) {
    runHostTests<short>(dev, ctxt, q, 4);
    runHostTests<int>(dev, ctxt, q, 42);
    runHostTests<long>(dev, ctxt, q, 424);
    runHostTests<long long>(dev, ctxt, q, 4242);
    if (HalfsSupported)
      runHostTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runHostTests<float>(dev, ctxt, q, 4.242f);
    if (DoublesSupported)
      runHostTests<double>(dev, ctxt, q, 4.24242);
    if (HalfsSupported && DoublesSupported)
      runHostTests<test_struct_all>(dev, ctxt, q, test_obj_all);
    else if (HalfsSupported)
      runHostTests<test_struct_half>(dev, ctxt, q, test_obj_half);
    else if (DoublesSupported)
      runHostTests<test_struct_double>(dev, ctxt, q, test_obj_double);
    else
      runHostTests<test_struct_minimum>(dev, ctxt, q, test_obj_minimum);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    runSharedTests<short>(dev, ctxt, q, 4);
    runSharedTests<int>(dev, ctxt, q, 42);
    runSharedTests<long>(dev, ctxt, q, 424);
    runSharedTests<long long>(dev, ctxt, q, 4242);
    if (HalfsSupported)
      runSharedTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runSharedTests<float>(dev, ctxt, q, 4.242f);
    if (DoublesSupported)
      runSharedTests<double>(dev, ctxt, q, 4.24242);
    if (HalfsSupported && DoublesSupported)
      runSharedTests<test_struct_all>(dev, ctxt, q, test_obj_all);
    else if (HalfsSupported)
      runSharedTests<test_struct_half>(dev, ctxt, q, test_obj_half);
    else if (DoublesSupported)
      runSharedTests<test_struct_double>(dev, ctxt, q, test_obj_double);
    else
      runSharedTests<test_struct_minimum>(dev, ctxt, q, test_obj_minimum);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    runDeviceTests<short>(dev, ctxt, q, 4);
    runDeviceTests<int>(dev, ctxt, q, 42);
    runDeviceTests<long>(dev, ctxt, q, 420);
    runDeviceTests<long long>(dev, ctxt, q, 4242);
    if (HalfsSupported)
      runDeviceTests<sycl::half>(dev, ctxt, q, sycl::half(4.2f));
    runDeviceTests<float>(dev, ctxt, q, 4.242f);
    if (DoublesSupported)
      runDeviceTests<double>(dev, ctxt, q, 4.24242);
    if (HalfsSupported && DoublesSupported)
      runDeviceTests<test_struct_all>(dev, ctxt, q, test_obj_all);
    else if (HalfsSupported)
      runDeviceTests<test_struct_half>(dev, ctxt, q, test_obj_half);
    else if (DoublesSupported)
      runDeviceTests<test_struct_double>(dev, ctxt, q, test_obj_double);
    else
      runDeviceTests<test_struct_minimum>(dev, ctxt, q, test_obj_minimum);
  }

  return 0;
}
