// REQUIRES: gpu
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
#include <iostream>
#include <sycl.hpp>
#include <vector>

using namespace sycl;

template <typename T> bool are_bitwise_equal(T lhs, T rhs) {
  constexpr size_t size{sizeof(T)};

  std::array<char, size> lhs_bytes;
  std::array<char, size> rhs_bytes;
  std::memcpy(lhs_bytes.data(), &lhs, size);
  std::memcpy(rhs_bytes.data(), &rhs, size);

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

  std::vector<T> result_source_vec(NumElems);
  std::vector<T> input_vec(NumElems);

  for (size_t i = 0; i < NumElems; ++i) {
    input_vec[i] = inexact * i;
  }
  {
    sycl::buffer<T> result_source_buf{result_source_vec};
    sycl::buffer<T> input_buf{input_vec};
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor out_source{result_source_buf, cgh};
      sycl::accessor in{input_buf, cgh};
      cgh.single_task<>([=]() {
        for (size_t i = 0; i < NumElems; ++i) {
          auto source = in[i];
          ++source;
          out_source[i] = source;
        }
      });
    });
    queue.wait_and_throw();
  } // buffers go out of scope here and write back to the vectors
  for (size_t i = 0; i < NumElems; ++i) {
    T expected_value = input_vec[i] + 1;

    if (!are_bitwise_equal(expected_value, result_source_vec[i])) {
      pass = false;
      std::cout << "Sample failed retrieved value: " << result_source_vec[i]
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
