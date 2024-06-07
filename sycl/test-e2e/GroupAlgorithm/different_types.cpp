// RUN: %{build} -fsycl-device-code-split=per_kernel -I . -o %t.out
// RUN: %{run} %t.out

#include "../helpers.hpp"
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/sycl_span.hpp>

using namespace sycl;

template <class OutT, class InputContainer, class InitT, class BinaryOperation>
void test(queue &q, const InputContainer &input, InitT init,
          BinaryOperation binary_op) {
  const int N = input.size();
  buffer b_in(input);
  buffer<OutT, 1> b_reduce_out(3);
  buffer<OutT, 2> b_scan_out(range(6, N));
  q.submit([&](handler &cgh) {
    accessor a_in(b_in, cgh, read_only);
    accessor a_reduce_out(b_reduce_out, cgh, write_only, no_init);
    accessor a_scan_out(b_scan_out, cgh, write_only, no_init);
    nd_range<1> r(N, N);
    cgh.parallel_for(r, [=](nd_item<1> it) {
      const auto g = it.get_group();
      const auto sg = it.get_sub_group();
      const auto idx = it.get_local_id();
      const auto begin = a_in.get_pointer();
      const auto end = a_in.get_pointer() + N;

      a_reduce_out[0] = reduce_over_group(g, a_in[idx], init, binary_op);
      a_reduce_out[1] = joint_reduce(g, begin, end, init, binary_op);
      a_reduce_out[2] = joint_reduce(sg, begin, end, init, binary_op);

      a_scan_out[0][idx] =
          exclusive_scan_over_group(g, a_in[idx], init, binary_op);
      joint_exclusive_scan(g, begin, end, &a_scan_out[1][0], init, binary_op);
      joint_exclusive_scan(sg, begin, end, &a_scan_out[2][0], init, binary_op);

      a_scan_out[3][idx] =
          inclusive_scan_over_group(g, a_in[idx], binary_op, init);
      joint_inclusive_scan(g, begin, end, &a_scan_out[4][0], binary_op, init);
      joint_inclusive_scan(sg, begin, end, &a_scan_out[5][0], binary_op, init);
    });
  });
  host_accessor a_reduce_out(b_reduce_out);
  host_accessor a_scan_out(b_scan_out);

  const auto r = std::accumulate(input.begin(), input.end(), init, binary_op);
  assert(r == a_reduce_out[0]);
  assert(r == a_reduce_out[1]);
  assert(r == a_reduce_out[2]);

  const auto equal = [](auto &&C1, auto &&C2) {
    return std::equal(C1.begin(), C1.end(), C2.begin());
  };
  std::vector<OutT> escan_ref(N);
  emu::exclusive_scan(input.begin(), input.end(), escan_ref.begin(), init,
                      binary_op);
  assert(equal(escan_ref, span(&a_scan_out[0][0], N)));
  assert(equal(escan_ref, span(&a_scan_out[1][0], N)));
  assert(equal(escan_ref, span(&a_scan_out[2][0], N)));

  std::vector<OutT> iscan_ref(N);
  emu::inclusive_scan(input.begin(), input.end(), iscan_ref.begin(), binary_op,
                      init);
  assert(equal(iscan_ref, span(&a_scan_out[3][0], N)));
  assert(equal(iscan_ref, span(&a_scan_out[4][0], N)));
  assert(equal(iscan_ref, span(&a_scan_out[5][0], N)));
}

int main() {
  using namespace std::complex_literals;
  queue q;
  auto repeat = [](auto val, int n) {
    return std::vector<decltype(val)>(n, val);
  };
  auto iota = [](auto val, int n) {
    std::vector<decltype(val)> v(n);
    std::iota(v.begin(), v.end(), val);
    return v;
  };
  constexpr int N = 64;
  auto u8max = std::numeric_limits<uint8_t>::max();
  auto fmax = std::numeric_limits<float>::max();
  uint8_t u8_1 = 1;
  test<int32_t>(q, repeat(u8_1, N), 0, sycl::plus<int32_t>());
  test<int32_t>(q, repeat(u8max, N), 0, sycl::plus<int32_t>());
  test<int32_t>(q, repeat(u8_1, N), 0, sycl::plus<>());
  test<int32_t>(q, repeat(u8max, N), 0, sycl::plus<>());
  test<int32_t>(q, iota(1.5f, N), 0, sycl::plus<int32_t>());
  test<float>(q, iota(1, N), 1.f, sycl::plus<float>());
  test<uint64_t>(q, iota(1, 15), uint64_t(0), sycl::multiplies<uint64_t>());
  test<std::complex<float>>(q, iota(1, 5), 1if,
                            sycl::plus<std::complex<float>>());
  if (q.get_device().has(aspect::fp64)) {
    test<double>(q, repeat(fmax, N), 0.0, sycl::plus<double>());
    test<std::complex<double>>(q, iota(1, 5), 1i,
                               sycl::plus<std::complex<double>>());
  }
  std::cout << "passed.\n";
  return 0;
}
