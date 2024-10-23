// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reduction_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

float sum(sycl::queue q, float *array, size_t N) {

  sycl::buffer<float> input_buf{array, N};
  sycl::buffer<float> result_buf{1};

  sycl::host_accessor{result_buf}[0] = 0;

  q.submit([&](sycl::handler &h) {
    auto input = sycl::accessor(input_buf, h, sycl::read_only);
    auto reduction = sycl::reduction(result_buf, h, sycl::plus<>(),
                                     syclex::properties(syclex::deterministic));
    h.parallel_for(N, reduction,
                   [=](size_t i, auto &reducer) { reducer += input[i]; });
  });

  return sycl::host_accessor{result_buf}[0];
}

int main(int argc, char *argv[]) {

  constexpr size_t N = 1024;
  std::array<float, N> array;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::generate(array.begin(), array.end(), [&]() { return dist(gen); });

  sycl::queue q;
  float x = sum(q, array.data(), N);
  float y = sum(q, array.data(), N);

  // NB: determinism guarantees bitwise reproducible reductions for floats
  assert(sycl::bit_cast<unsigned int>(x) == sycl::bit_cast<unsigned int>(y));
}
