#ifndef EXCLUSIVE_SCAN_OVER_GROUP_HPP
#define EXCLUSIVE_SCAN_OVER_GROUP_HPP

#include <array>
#include <numeric>

#include "../../sycl_complex_helper.hpp"

////////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename BinaryOperation>
bool exclusive_scan_over_group(sycl::queue q, T input,
                               BinaryOperation binary_op) {
  using V = typename T::value_type;

  bool result = true;

  constexpr size_t N = input.size();

  auto init = sycl::ext::oneapi::experimental::cplx::detail::get_init<
      V, BinaryOperation>();

  auto *in = sycl::malloc_shared<V>(N, q);
  auto *output_with_init = sycl::malloc_shared<V>(N, q);
  auto *output_without_init = sycl::malloc_shared<V>(N, q);

  for (std::size_t i = 0; i < N; i++) {
    in[i] = input[i];
  }

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id(0);
      auto lid = it.get_local_id(0);
      auto g = it.get_group();

      output_with_init[lid] =
          sycl::ext::oneapi::experimental::exclusive_scan_over_group(
              g, in[gid], init, binary_op);
      output_without_init[lid] =
          sycl::ext::oneapi::experimental::exclusive_scan_over_group(g, in[gid],
                                                                     binary_op);
    });
  });

  q.wait();

  std::array<V, N> expected;
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init,
                      binary_op);

  std::array<V, N> output;
  for (std::size_t i = 0; i < N; i++) {
    output[i] = output_with_init[i];
  }

  result &= check_results(output, expected, true);

  for (std::size_t i = 0; i < N; i++) {
    output[i] = output_without_init[i];
  }

  result &= check_results(output, expected, true);

  sycl::free(in, q);
  sycl::free(output_with_init, q);
  sycl::free(output_without_init, q);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// COMPLEX TESTS
////////////////////////////////////////////////////////////////////////////////

template <typename T, std::size_t N, typename BinaryOperation>
bool test_scalar_exclusive_scan_over_group() {
  using Complex = typename sycl::ext::oneapi::experimental::complex<T>;
  using Array = typename std::array<Complex, N>;

  bool result = true;

  sycl::queue q;

  const auto test_cases = std::array<Array, 7>{
      // Basic value test
      Array{Complex{1, 0}, Complex{2, 0}, Complex{3, 0}, Complex{4, 0}},
      // Random value test
      Array{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
            Complex{3.7, 3.7}},
      // Repeated value test
      Array{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
      // Negative value test
      Array{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
            Complex{0, 0}},
      // Large value test
      Array{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
            Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
      // Small value test
      Array{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
            Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
      // Edge case value test
      Array{Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
            Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}}};
  const auto binary_op = BinaryOperation{};

  if (is_type_supported<T>(q)) {
    for (const auto &test_case : test_cases) {
      result &= exclusive_scan_over_group(q, test_case, binary_op);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// MARRAY<COMPLEX> TESTS
////////////////////////////////////////////////////////////////////////////////

template <typename T, std::size_t N, typename BinaryOperation>
bool test_marray_exclusive_scan_over_group() {
  using Complex = typename sycl::ext::oneapi::experimental::complex<T>;
  using Marray = typename sycl::marray<Complex, N>;
  using Array = typename std::array<Marray, N>;

  bool result = true;

  sycl::queue q;

  const auto test_cases = std::array<Array, 7>{
      // Basic value test
      Array{Marray{Complex{1, 0}, Complex{1, 0}, Complex{1, 0}, Complex{1, 0}},
            Marray{Complex{2, 0}, Complex{2, 0}, Complex{2, 0}, Complex{2, 0}},
            Marray{Complex{3, 0}, Complex{3, 0}, Complex{3, 0}, Complex{3, 0}},
            Marray{Complex{4, 0}, Complex{4, 0}, Complex{4, 0}, Complex{4, 0}}},
      // Random value test
      Array{Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}},
            Marray{Complex{0.5, 0.5}, Complex{1.2, 1.2}, Complex{-2.8, -2.8},
                   Complex{3.7, 3.7}}},
      // Repeated value test
      Array{Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}},
            Marray{Complex{1, 1}, Complex{1, 1}, Complex{1, 1}, Complex{1, 1}}},
      // Negative value test
      Array{Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}},
            Marray{Complex{-3.0, -3.0}, Complex{2.5, 2.5}, Complex{-1.2, -1.2},
                   Complex{0, 0}}},
      // Large value test
      Array{
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}},
          Marray{Complex{1000000.0, 1000000.0}, Complex{2000000.0, 2000000.0},
                 Complex{3000000.0, 3000000.0}, Complex{4000000.0, 4000000.0}}},
      // Small value test
      Array{Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}},
            Marray{Complex{0.0001, 0.0001}, Complex{0.0002, 0.0002},
                   Complex{0.0003, 0.0003}, Complex{0.0004, 0.0004}}},
      // Edge case value test
      Array{
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{
              Complex{nan_val<T>, nan_val<T>}, Complex{inf_val<T>, inf_val<T>},
              Complex{nan_val<T>, inf_val<T>}, Complex{inf_val<T>, nan_val<T>}},
          Marray{Complex{nan_val<T>, nan_val<T>},
                 Complex{inf_val<T>, inf_val<T>},
                 Complex{nan_val<T>, inf_val<T>},
                 Complex{inf_val<T>, nan_val<T>}}}};
  const auto binary_op = BinaryOperation{};

  if (is_type_supported<T>(q)) {
    for (const auto &test_case : test_cases) {
      result &= exclusive_scan_over_group(q, test_case, binary_op);
    }
  }

  return result;
}

#endif