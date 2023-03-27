//==---- copy.cpp - USM copy test ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::usm;

template <typename T> class transfer;

static constexpr int N = 100; // should be even

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

template <typename T> T *regular(queue q, alloc kind) {
  return malloc<T>(N, q, kind);
}

template <typename T> T *aligned(queue q, alloc kind) {
  return aligned_alloc<T>(alignof(long long), N, q, kind);
}

template <typename T> void test(queue q, T val, T *src, T *dst, bool dev_dst) {
  if (std::is_same_v<T, double> && !q.get_device().has(aspect::fp64))
    return;

  q.fill(src, val, N).wait();

  // Use queue::copy for the first half and handler::copy for the second
  q.copy(src, dst, N / 2).wait();
  q.submit([&](handler &h) { h.copy(src + N / 2, dst + N / 2, N / 2); }).wait();

  T *out = dst;

  std::array<T, N> arr;
  if (dev_dst) { // if copied to device, transfer data back to host
    buffer buf{arr};
    q.submit([&](handler &h) {
      accessor acc{buf, h};
      h.parallel_for<transfer<T>>(N, [=](id<1> i) { acc[i] = dst[i]; });
    });
    out = arr.data();
  }

  for (int i = 0; i < N; ++i) {
    assert(out[i] == val);
  }

  free(src, q);
  free(dst, q);
}

template <typename T> void runTests(queue q, T val, alloc kind1, alloc kind2) {
  bool dev_dst1 = (kind1 == alloc::device);
  bool dev_dst2 = (kind2 == alloc::device);
  test(q, val, regular<T>(q, kind1), regular<T>(q, kind2), dev_dst2);
  test(q, val, regular<T>(q, kind2), regular<T>(q, kind1), dev_dst1);
  test(q, val, aligned<T>(q, kind1), aligned<T>(q, kind2), dev_dst2);
  test(q, val, aligned<T>(q, kind2), aligned<T>(q, kind1), dev_dst1);
  test(q, val, regular<T>(q, kind1), aligned<T>(q, kind2), dev_dst2);
  test(q, val, regular<T>(q, kind2), aligned<T>(q, kind1), dev_dst1);
  test(q, val, aligned<T>(q, kind1), regular<T>(q, kind2), dev_dst2);
  test(q, val, aligned<T>(q, kind2), regular<T>(q, kind1), dev_dst1);
}

int main() {
  queue q;
  auto dev = q.get_device();

  const bool DoublesSupported = dev.has(sycl::aspect::fp64);
  const bool HalfsSupported = dev.has(sycl::aspect::fp16);

  test_struct_all test_obj_all{4, 42, 424, 4242, 4.2f, 4.242, 4.24242};
  test_struct_half test_obj_half{4, 42, 424, 4242, 4.2f, 4.242};
  test_struct_double test_obj_double{4, 42, 424, 4242, 4.242, 4.24242};
  test_struct_minimum test_obj_minimum{4, 42, 424, 4242, 4.242};

  if (dev.has(aspect::usm_host_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::host);
    runTests<int>(q, 42, alloc::host, alloc::host);
    runTests<long>(q, 424, alloc::host, alloc::host);
    runTests<long long>(q, 4242, alloc::host, alloc::host);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::host, alloc::host);
    runTests<float>(q, 4.242f, alloc::host, alloc::host);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::host, alloc::host);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::host, alloc::host);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::host, alloc::host);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::host,
                                   alloc::host);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::host,
                                    alloc::host);
  }

  if (dev.has(aspect::usm_shared_allocations)) {
    runTests<short>(q, 4, alloc::shared, alloc::shared);
    runTests<int>(q, 42, alloc::shared, alloc::shared);
    runTests<long>(q, 424, alloc::shared, alloc::shared);
    runTests<long long>(q, 4242, alloc::shared, alloc::shared);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::shared, alloc::shared);
    runTests<float>(q, 4.242f, alloc::shared, alloc::shared);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::shared, alloc::shared);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::shared, alloc::shared);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::shared,
                                 alloc::shared);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::shared,
                                   alloc::shared);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::shared,
                                    alloc::shared);
  }

  if (dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::device, alloc::device);
    runTests<int>(q, 42, alloc::device, alloc::device);
    runTests<long>(q, 424, alloc::device, alloc::device);
    runTests<long long>(q, 4242, alloc::device, alloc::device);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::device, alloc::device);
    runTests<float>(q, 4.242f, alloc::device, alloc::device);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::device, alloc::device);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::device, alloc::device);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::device,
                                 alloc::device);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::device,
                                   alloc::device);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::device,
                                    alloc::device);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_shared_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::shared);
    runTests<int>(q, 42, alloc::host, alloc::shared);
    runTests<long>(q, 424, alloc::host, alloc::shared);
    runTests<long long>(q, 4242, alloc::host, alloc::shared);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::host, alloc::shared);
    runTests<float>(q, 4.242f, alloc::host, alloc::shared);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::host, alloc::shared);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::host, alloc::shared);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::host, alloc::shared);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::host,
                                   alloc::shared);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::host,
                                    alloc::shared);
  }

  if (dev.has(aspect::usm_host_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::host, alloc::device);
    runTests<int>(q, 42, alloc::host, alloc::device);
    runTests<long>(q, 424, alloc::host, alloc::device);
    runTests<long long>(q, 4242, alloc::host, alloc::device);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::host, alloc::device);
    runTests<float>(q, 4.242f, alloc::host, alloc::device);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::host, alloc::device);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::host, alloc::device);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::host, alloc::device);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::host,
                                   alloc::device);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::host,
                                    alloc::device);
  }

  if (dev.has(aspect::usm_shared_allocations) &&
      dev.has(aspect::usm_device_allocations)) {
    runTests<short>(q, 4, alloc::shared, alloc::device);
    runTests<int>(q, 42, alloc::shared, alloc::device);
    runTests<long>(q, 424, alloc::shared, alloc::device);
    runTests<long long>(q, 4242, alloc::shared, alloc::device);
    if (HalfsSupported)
      runTests<half>(q, half(4.2f), alloc::shared, alloc::device);
    runTests<float>(q, 4.242f, alloc::shared, alloc::device);
    if (DoublesSupported)
      runTests<double>(q, 4.24242, alloc::shared, alloc::device);
    if (HalfsSupported && DoublesSupported)
      runTests<test_struct_all>(q, test_obj_all, alloc::shared, alloc::device);
    else if (HalfsSupported)
      runTests<test_struct_half>(q, test_obj_half, alloc::shared,
                                 alloc::device);
    else if (DoublesSupported)
      runTests<test_struct_double>(q, test_obj_double, alloc::shared,
                                   alloc::device);
    else
      runTests<test_struct_minimum>(q, test_obj_minimum, alloc::shared,
                                    alloc::device);
  }

  return 0;
}
