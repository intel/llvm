// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include <algorithm>
#include <complex>
#include <numeric>

#include <sycl/detail/core.hpp>

#include <sycl/reduction.hpp>

using namespace sycl;

#define BUFFER_SIZE 255

// Currently, Identityless reduction for complex numbers is
// only valid for plus operator.
// TODO: Extend this test case once we support known_identity for std::complex
// and more operators (apart from plus).
template <typename T>
void test_identityless_reduction_for_complex_nums(queue &q) {
  // Allocate and initialize buffer on the host with all 1's.
  buffer<std::complex<T>> valuesBuf{BUFFER_SIZE};
  {
    host_accessor a{valuesBuf};
    T n = 0;
    std::generate(a.begin(), a.end(), [&n] {
      n++;
      return std::complex<T>(n, n + 1);
    });
  }

  // Buffer to hold the reduction results.
  std::complex<T> sumResult = 0;
  buffer<std::complex<T>> sumBuf{&sumResult, 1};

  q.submit([&](handler &cgh) {
    accessor inputVals{valuesBuf, cgh, sycl::read_only};
    auto sumReduction = reduction(sumBuf, cgh, plus<std::complex<T>>());

    cgh.parallel_for(nd_range<1>{BUFFER_SIZE, BUFFER_SIZE}, sumReduction,
                     [=](nd_item<1> idx, auto &sum) {
                       sum += inputVals[idx.get_global_id(0)];
                     });
  });

  assert(sumBuf.get_host_access()[0] == std::complex<T>(32640, 32895));
}

int main() {
  queue q;

  test_identityless_reduction_for_complex_nums<float>(q);
  if (q.get_device().has(aspect::fp64))
    test_identityless_reduction_for_complex_nums<double>(q);

  return 0;
}
