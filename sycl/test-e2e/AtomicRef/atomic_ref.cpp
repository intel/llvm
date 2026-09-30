// REQUIRES: aspect-ext_oneapi_atomic16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/usm.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>

using namespace sycl;

// Number of work-items contending on the same atomic location.
constexpr size_t N = 64;

// Compares result against expected (with a tolerance for sycl::half), frees
// the USM allocation, and reports 0 on success or 1 on failure.
template <typename T> int CheckResult(T result, T expected, T *data, queue &q) {
  bool passed;
  if constexpr (std::is_same_v<T, half>) {
    passed = std::fabs(static_cast<float>(result) -
                       static_cast<float>(expected)) < 0.001f;
  } else {
    passed = (result == expected);
  }

  free(data, q);

  if (!passed) {
    std::cerr << "CheckResult FAILED: expected " << static_cast<float>(expected)
              << " but got " << static_cast<float>(result) << std::endl;
    return 1;
  }

  return 0;
}

template <typename T> int test_atomic_sub() {
  queue q;
  T *data = malloc_shared<T>(1, q);

  // initial must stay comfortably above N * decrement so unsigned types
  // (e.g. unsigned short) don't underflow, and both initial and the
  // running total must stay within bfloat16's exact-integer range (<=256).
  const T initial = static_cast<T>(100);
  const T decrement = static_cast<T>(1);
  *data = initial;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1>) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       atomic_data.fetch_sub(decrement);
     });
   }).wait();

  const T result = *data;
  const T expected = static_cast<T>(static_cast<float>(initial) -
                                    N * static_cast<float>(decrement));

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_add() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  // Keep initial + N * increment within bfloat16's exact-integer range
  // (<=256).
  const T initial = static_cast<T>(10);
  const T increment = static_cast<T>(1);
  *data = initial;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1>) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       atomic_data.fetch_add(increment);
     });
   }).wait();

  const T result = *data;
  const T expected = static_cast<T>(static_cast<float>(initial) +
                                    N * static_cast<float>(increment));

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_min() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  const T initial = static_cast<T>(10);
  const T other = static_cast<T>(3);
  *data = initial;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1>) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       atomic_data.fetch_min(other);
     });
   }).wait();

  const T result = *data;
  const T expected =
      static_cast<float>(initial) < static_cast<float>(other) ? initial : other;

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_max() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  const T initial = static_cast<T>(10);
  const T other = static_cast<T>(3);
  *data = initial;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1>) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       atomic_data.fetch_max(other);
     });
   }).wait();

  const T result = *data;
  const T expected =
      static_cast<float>(initial) > static_cast<float>(other) ? initial : other;

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_or() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  const T initial = static_cast<T>(0);
  *data = initial;

  // Each work-item ORs in a distinct bit (cycling through all bits of T), so
  // full coverage across N work-items forces real contention on the CAS
  // retry loop while still producing a deterministic expected result: every
  // bit of T ends up set.
  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1> it) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       const T bit = static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8)));
       atomic_data.fetch_or(bit);
     });
   }).wait();

  const T result = *data;
  const T expected = static_cast<T>(~static_cast<T>(0));

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_and() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  const T initial = static_cast<T>(~static_cast<T>(0));
  *data = initial;

  // Each work-item ANDs out a distinct bit (cycling through all bits of T),
  // so full coverage across N work-items forces real contention on the CAS
  // retry loop while still producing a deterministic expected result: every
  // bit of T ends up cleared.
  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1> it) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       const T mask = static_cast<T>(
           ~static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8))));
       atomic_data.fetch_and(mask);
     });
   }).wait();

  const T result = *data;
  const T expected = static_cast<T>(0);

  return CheckResult(result, expected, data, q);
}

template <typename T> int test_atomic_xor() {
  queue q;

  T *data = malloc_shared<T>(1, q);

  const T initial = static_cast<T>(0);
  *data = initial;

  // Each work-item XORs a distinct bit (cycling through all bits of T), so
  // full coverage across N work-items forces real contention on the CAS
  // retry loop. N is an exact multiple of the bit width, so every bit gets
  // toggled the same (even) number of times, giving a deterministic
  // expected result: every bit ends up back at its initial value.
  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1> it) {
       atomic_ref<T, memory_order::relaxed, memory_scope::device,
                  access::address_space::global_space>
           atomic_data(*data);
       const T bit = static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8)));
       atomic_data.fetch_xor(bit);
     });
   }).wait();

  const T result = *data;
  const T expected = initial;

  return CheckResult(result, expected, data, q);
}

int main() {

  int ret = test_atomic_sub<sycl::half>();
  ret |= test_atomic_sub<unsigned short>();
  ret |= test_atomic_sub<short>();
  ret |= test_atomic_sub<sycl::ext::oneapi::bfloat16>();

  ret |= test_atomic_add<sycl::half>();
  ret |= test_atomic_add<unsigned short>();
  ret |= test_atomic_add<short>();
  ret |= test_atomic_add<sycl::ext::oneapi::bfloat16>();

  ret |= test_atomic_min<sycl::half>();
  ret |= test_atomic_min<unsigned short>();
  ret |= test_atomic_min<short>();
  ret |= test_atomic_min<sycl::ext::oneapi::bfloat16>();

  ret |= test_atomic_max<sycl::half>();
  ret |= test_atomic_max<unsigned short>();
  ret |= test_atomic_max<short>();
  ret |= test_atomic_max<sycl::ext::oneapi::bfloat16>();

  ret |= test_atomic_or<unsigned short>();
  ret |= test_atomic_or<short>();

  ret |= test_atomic_and<unsigned short>();
  ret |= test_atomic_and<short>();

  ret |= test_atomic_xor<unsigned short>();
  ret |= test_atomic_xor<short>();

  return ret;
}
