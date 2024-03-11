// Temporarily add explicit '-O2' to avoid GPU hang issue with O0 optimization.
// RUN: %{build} -fno-sycl-id-queries-fit-in-int -O2 -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<int> C;
int n_fail = 0;

template <typename ContainerT>
void check_sum(std::string_view desc, const ContainerT &data, size_t N) {
  auto res = std::accumulate(data.begin(), data.end(), size_t(0));
  if (res != N) {
    std::cout << desc << " fail\n"
              << "got:      " << res << "\n"
              << "expected: " << N << "\n";
    ++n_fail;
  } else {
    std::cout << desc << " pass\n";
  }
}

template <typename RangeT>
void test_regular(std::string_view desc, queue &q, size_t B, RangeT range) {
  auto N = range.size();
  std::vector accumulators_v(B, 0);
  {
    sycl::buffer accumulator_buf{accumulators_v};
    q.submit([&](sycl::handler &h) {
      sycl::accessor accumulators{accumulator_buf, h};
      h.parallel_for(range, [=](auto it) {
        atomic_ref<int, memory_order::relaxed, memory_scope::device> ref(
            accumulators[it.get_linear_id() % B]);
        ++ref;
      });
    });
  } // destruction of accumulator_buf here writes back data to accumulators_v
  check_sum(desc, accumulators_v, N);
}

template <typename RangeT>
void test_spec_constant(std::string_view desc, queue &q, size_t B,
                        RangeT range) {
  auto N = range.size();
  std::vector accumulators_v(B, 0);
  {
    sycl::buffer accumulators_buf{accumulators_v};
    q.submit([&](handler &cgh) {
      sycl::accessor accumulators{accumulators_buf, cgh};
      cgh.set_specialization_constant<C>(2);
      cgh.parallel_for(range, [=](auto it, kernel_handler h) {
        atomic_ref<int, memory_order::relaxed, memory_scope::device> ref(
            accumulators[it.get_linear_id() % B]);
        ref += h.get_specialization_constant<C>();
      });
    });
  } // destruction of accumulators_buf here writes data back to accumulators_v
  check_sum(desc, accumulators_v, N * 2);
}

int main(int argc, char *argv[]) {
  if (sizeof(size_t) <= 4) {
    std::cout << "size_t is 32-bit, nothing to do\n";
    return 0;
  }

  queue q;
  constexpr int B = 1000000;
  // First prime bigger than UINT32_MAX
  constexpr size_t N = 4'294'967'311;
  test_regular("regular range<1>", q, B, range(N));
  test_regular("regular range<2>", q, B, range(1, N));
  test_regular("regular range<3>", q, B, range(1, 1, N));
  test_spec_constant("spec constant range<1>", q, B, range(N));
  test_spec_constant("spec constant range<2>", q, B, range(N, 1));
  test_spec_constant("spec constant range<3>", q, B, range(N, 1, 1));

  try {
    q.parallel_for(range(std::numeric_limits<size_t>::max(), 2),
                   [](auto id) {});
  } catch (sycl::exception &e) {
    assert(e.code() == errc::runtime);
  }

  return n_fail != 0;
}
