// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out
//
// The test fails on opencl:cpu when running on AMD runner and passes when
// running on Intel Arc GPU runner.
// UNSUPPORTED: cpu

// Test fusion works with reductions.

#include <sycl/sycl.hpp>

#include "../helpers.hpp"

using namespace sycl;

template <typename BinaryOperation> class ReductionTest;

int main() {
  constexpr size_t dataSize = 512;

  int sumRes = -1;
  int maxRes = -1;

  {
    queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

    buffer<int> dataBuf{dataSize};
    buffer<int> sumBuf{&sumRes, 1};
    buffer<int> maxBuf{&maxRes, 1};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, dataBuf, 0);

    q.submit([&](handler &cgh) {
      accessor in(dataBuf, cgh, read_only);
      auto sumRed = reduction(sumBuf, cgh, plus<>{},
                              property::reduction::initialize_to_identity{});
      auto maxRed = reduction(maxBuf, cgh, maximum<>{},
                              property::reduction::initialize_to_identity{});
      cgh.parallel_for(dataSize, sumRed, maxRed,
                       [=](id<1> i, auto &sum, auto &max) {
                         sum.combine(in[i]);
                         max.combine(in[i]);
                       });
    });

    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }

  constexpr int expectedMax = dataSize - 1;
  constexpr int expectedSum = dataSize * expectedMax / 2;

  std::cerr << sumRes << "\n";

  assert(maxRes == expectedMax && "Unexpected max value");
  assert(sumRes == expectedSum && "Unexpected sum value");

  return 0;
}
