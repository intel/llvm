// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel -fsycl-unnamed-lambda %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: gpu
#include <iostream>
#include <sycl.hpp>
#include <vector>

using namespace sycl;

template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T> using shared_vector = std::vector<T, shared_allocator<T>>;

template <typename T> bool are_bitwise_equal(T lhs, T rhs) {
  constexpr size_t size{sizeof(T)};

  // Such type-punning is OK from the point of strict aliasing rules
  const auto &lhs_bytes = reinterpret_cast<const unsigned char(&)[size]>(lhs);
  const auto &rhs_bytes = reinterpret_cast<const unsigned char(&)[size]>(rhs);

  bool result{true};
  for (size_t i = 0; i < size; ++i) {
    result &= lhs_bytes[i] == rhs_bytes[i];
  }
  return result;
}

template <typename T> bool test(sycl::queue &queue) {
  constexpr int NumElems{32};
  bool pass{true};

  static const T inexact = static_cast<T>(0.1);

  shared_vector<T> result_source{NumElems, shared_allocator<T>{queue}};
  shared_vector<T> input{NumElems, shared_allocator<T>{queue}};

  for (size_t i = 0; i < NumElems; ++i) {
    input[i] = inexact * i;
  }

  queue.submit([&](sycl::handler &cgh) {
    auto out_source = result_source.data();
    auto in = input.data();

    cgh.single_task<>([=]() {
      for (size_t i = 0; i < NumElems; ++i) {
        auto source = in[i];
        ++source;
        out_source[i] = source;
      }
    });
  });
  queue.wait_and_throw();

  for (size_t i = 0; i < NumElems; ++i) {
    T expected_value = input[i] + 1;

    if (!are_bitwise_equal(expected_value, result_source[i])) {
      pass = false;
      std::cout << "Sample failed retrieved value: " << result_source[i]
                << ", but expected: " << expected_value << ", at index: " << i
                << std::endl;
    }
  }
  return pass;
}

int main(int argc, char **argv) {
  sycl::queue queue{};
  bool passed = true;
  passed &= test<float>(queue);
  if (queue.get_device().has(sycl::aspect::fp16))
    passed &= test<sycl::half>(queue);
  return passed ? 0 : 1;
}
