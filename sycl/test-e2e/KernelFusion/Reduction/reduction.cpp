// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

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

    complete_fusion_with_check(fw);
  }

  constexpr int expectedMax = dataSize - 1;
  constexpr int expectedSum = dataSize * expectedMax / 2;

  assert(maxRes == expectedMax && "Unexpected max value");
  assert(sumRes == expectedSum && "Unexpected sum value");

  return 0;
}
