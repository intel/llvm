// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "support.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
using namespace sycl;

template <typename T> bool equal(const T &a, const T &b) { return a == b; }

template <typename T, int N>
bool equal(const vec<T, N> &a, const vec<T, N> &b) {
  for (int i = 0; i < N; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

template <typename kernel_name, typename InputContainer,
          typename OutputContainer>
void test(queue q, InputContainer input, OutputContainer output) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  size_t N = input.size();
  size_t G = 4;
  range<2> R(G, G);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name>(nd_range<2>(R, R), [=](nd_item<2> it) {
        group<2> g = it.get_group();
        int lid = it.get_local_linear_id();
        out[0] = group_broadcast(g, in[lid]);
        out[1] = group_broadcast(g, in[lid], group<2>::id_type(1, 2));
        out[2] =
            group_broadcast(g, in[lid], group<2>::linear_id_type(2 * G + 1));
      });
    });
  }
  assert(equal(output[0], input[0]));
  assert(equal(output[1], input[1 * G + 2]));
  assert(equal(output[2], input[2 * G + 1]));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 16;

  // Test built-in scalar type
  {
    std::array<int, N> input;
    std::array<int, 3> output;
    std::iota(input.begin(), input.end(), 1);
    std::fill(output.begin(), output.end(), false);
    test<class KernelName_EFL>(q, input, output);
  }

  // Test pointer types
  {
    std::array<int *, N> input;
    std::array<int *, 3> output;
    for (int i = 0; i < N; ++i) {
      input[i] = static_cast<int *>(0x0) + i;
    }
    std::fill(output.begin(), output.end(), static_cast<int *>(0x0));
    test<class KernelName_NrqELzFQToOSPsRNMi>(q, input, output);
  }

  // Test vector types
  {
    std::array<vec<int, 4>, N> input;
    std::array<vec<int, 4>, 3> output;
    for (int i = 0; i < N; ++i) {
      input[i] = vec<int, 4>{i, i, i, i};
    }
    std::fill(output.begin(), output.end(), vec<int, 4>{0, 0, 0, 0});
    test<class KernelName_VectorGroupBroadcast>(q, input, output);
  }

  // Test user-defined type
  // - Use complex as a proxy for this
  // - Test float and double to test 64-bit and 128-bit types
  {
    std::array<std::complex<float>, N> input;
    std::array<std::complex<float>, 3> output;
    for (int i = 0; i < N; ++i) {
      input[i] =
          std::complex<float>(0, 1) + (float)i * std::complex<float>(2, 2);
    }
    std::fill(output.begin(), output.end(), std::complex<float>(0, 0));
    test<class KernelName_rCblcml>(q, input, output);
  }
  if (q.get_device().has(sycl::aspect::fp64)) {
    std::array<std::complex<double>, N> input;
    std::array<std::complex<double>, 3> output;
    for (int i = 0; i < N; ++i) {
      input[i] =
          std::complex<double>(0, 1) + (double)i * std::complex<double>(2, 2);
    }
    std::fill(output.begin(), output.end(), std::complex<float>(0, 0));
    test<class KernelName_NCWhjnQ>(q, input, output);
  }
  std::cout << "Test passed." << std::endl;
}
