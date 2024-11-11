// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reduction_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int sum(sycl::queue q, int *array, size_t N) {

  sycl::buffer<int> input_buf{array, N};
  sycl::buffer<int> result_buf{1};

  sycl::host_accessor{result_buf}[0] = 42;

  q.submit([&](sycl::handler &h) {
    auto input = sycl::accessor(input_buf, h, sycl::read_only);
    auto reduction =
        sycl::reduction(result_buf, h, sycl::plus<>(),
                        syclex::properties(syclex::initialize_to_identity));
    h.parallel_for(N, reduction,
                   [=](size_t i, auto &reducer) { reducer += input[i]; });
  });

  return sycl::host_accessor{result_buf}[0];
}

int main(int argc, char *argv[]) {

  constexpr size_t N = 32;
  std::array<int, N> array;
  std::iota(array.begin(), array.end(), 1);

  sycl::queue q;
  int x = sum(q, array.data(), N);
  assert(x == 528);
}
