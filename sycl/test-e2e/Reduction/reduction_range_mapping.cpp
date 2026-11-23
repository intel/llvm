// RUN: %{build} -D__SYCL_REDUCTION_NUM_CONCURRENT_WORKGROUPS=3 -o %t.out
// RUN: %{run} %t.out

// Check that every range element is processed exactly once and that reductions
// produce the expected count and sum. Include groups processing multiple blocks
// and an incomplete final block.
#include <sycl/detail/core.hpp>
#include <sycl/reduction.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

template <int Dims> void test(sycl::queue &Q, sycl::range<Dims> Range) {
  const size_t N = Range.size();
  std::vector<unsigned> Visits(N, 0);
  unsigned Count = 19, Sum = 23;
  {
    sycl::buffer<unsigned> Visited(Visits.data(), sycl::range<1>(N));
    sycl::buffer<unsigned> CountBuf(&Count, sycl::range<1>(1));
    sycl::buffer<unsigned> SumBuf(&Sum, sycl::range<1>(1));
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor V(Visited, CGH);
      auto C =
          sycl::reduction(CountBuf, CGH, sycl::plus<unsigned>(),
                          sycl::property::reduction::initialize_to_identity{});
      auto S = sycl::reduction(SumBuf, CGH, sycl::plus<unsigned>());
      CGH.parallel_for(Range, C, S,
                       [=](sycl::item<Dims> Item, auto &C, auto &S) {
                         size_t Linear = Item.get_linear_id();
                         assert(Linear < N);
                         sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                          sycl::memory_scope::device>
                             Visit(V[Linear]);
                         Visit.fetch_add(1);
                         C += 1u;
                         S += unsigned(Linear % 251);
                       });
    });
    Q.wait_and_throw();
  }
  assert(std::all_of(Visits.begin(), Visits.end(),
                     [](unsigned V) { return V == 1; }));
  unsigned ExpectedSum = 23;
  for (size_t I = 0; I < N; ++I)
    ExpectedSum += I % 251;
  assert(Count == N);
  assert(Sum == ExpectedSum);
}

int main() {
  sycl::queue Q;
  size_t WGSize = 0;
  Q.submit([&](sycl::handler &CGH) {
    WGSize = sycl::detail::reduGetPreferredWGSize(CGH, 2 * sizeof(unsigned));
  });
  for (size_t N : {24 * WGSize - 1, 24 * WGSize, 24 * WGSize + 1,
                   40 * WGSize + 1, 72 * WGSize + 17})
    test(Q, sycl::range<1>(N));
  test(Q, sycl::range<2>(7, 9 * WGSize + 1));
  test(Q, sycl::range<3>(3, 7, 5 * WGSize + 1));
}
