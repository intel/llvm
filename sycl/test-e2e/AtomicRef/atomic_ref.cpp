// REQUIRES: arch-intel_gpu_cri

// RUN: %{build}  -Xclang -freg-struct-return -Xspirv-translator=spir64 --spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support

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

template <typename T>
using AtomicRefT = atomic_ref<T, memory_order::relaxed, memory_scope::device,
                              access::address_space::global_space>;

template <typename T> int CheckResult(T result, T expected, T *data, queue &q) {
  bool passed;
  if constexpr (std::is_same_v<T, half>) {
    passed = std::fabs(static_cast<float>(result) -
                       static_cast<float>(expected)) < 0.001f;
  } else {
    passed = (result == expected);
  }

  if (!passed) {
    std::cerr << "CheckResult FAILED: expected " << static_cast<float>(expected)
              << " but got " << static_cast<float>(result) << std::endl;
    return 1;
  }

  return 0;
}

template <typename T, typename OperandGen, typename ApplyOp>
int test_atomic(T initial, OperandGen genOperand, ApplyOp applyOp, T expected) {
  queue q;
  T *data = malloc_shared<T>(1, q);
  *data = initial;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(range<1>(N), [=](item<1> it) {
       AtomicRefT<T> atomic_data(*data);
       applyOp(atomic_data, genOperand(it));
     });
   }).wait();

  const T result = *data;
  auto ret = CheckResult(result, expected, data, q);
  free(data, q);
  return ret;
}

template <typename T> int test_atomic_sub() {
  const T initial = static_cast<T>(100);
  const T decrement = static_cast<T>(1);
  const T expected = static_cast<T>(static_cast<float>(initial) -
                                    N * static_cast<float>(decrement));

  return test_atomic(
      initial, [=](item<1>) { return decrement; },
      [](auto &a, auto v) { a.fetch_sub(v); }, expected);
}

template <typename T> int test_atomic_add() {
  const T initial = static_cast<T>(10);
  const T increment = static_cast<T>(1);
  const T expected = static_cast<T>(static_cast<float>(initial) +
                                    N * static_cast<float>(increment));

  return test_atomic(
      initial, [=](item<1>) { return increment; },
      [](auto &a, auto v) { a.fetch_add(v); }, expected);
}

template <typename T> int test_atomic_min() {
  const T initial = static_cast<T>(10);
  const T other = static_cast<T>(3);
  const T expected =
      static_cast<float>(initial) < static_cast<float>(other) ? initial : other;

  return test_atomic(
      initial, [=](item<1>) { return other; },
      [](auto &a, auto v) { a.fetch_min(v); }, expected);
}

template <typename T> int test_atomic_max() {
  const T initial = static_cast<T>(10);
  const T other = static_cast<T>(3);
  const T expected =
      static_cast<float>(initial) > static_cast<float>(other) ? initial : other;

  return test_atomic(
      initial, [=](item<1>) { return other; },
      [](auto &a, auto v) { a.fetch_max(v); }, expected);
}

template <typename T> int test_atomic_or() {
  const T initial = static_cast<T>(0);
  const T expected = static_cast<T>(~static_cast<T>(0));

  return test_atomic(
      initial,
      [](item<1> it) {
        return static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8)));
      },
      [](auto &a, auto v) { a.fetch_or(v); }, expected);
}

template <typename T> int test_atomic_and() {
  const T initial = static_cast<T>(~static_cast<T>(0));
  const T expected = static_cast<T>(0);

  return test_atomic(
      initial,
      [](item<1> it) {
        return static_cast<T>(
            ~static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8))));
      },
      [](auto &a, auto v) { a.fetch_and(v); }, expected);
}

template <typename T> int test_atomic_xor() {
  const T initial = static_cast<T>(0);
  const T expected = initial;

  return test_atomic(
      initial,
      [](item<1> it) {
        return static_cast<T>(1 << (it.get_id(0) % (sizeof(T) * 8)));
      },
      [](auto &a, auto v) { a.fetch_xor(v); }, expected);
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
