// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: cuda || hip

// This test checks the spenario of using specialization constants with an
// 'array of array' as well as a 'stuct with an array of array' types for
// vector convolution as it is described in chapter 4.9.5. Specialization
// constants of the SYCL 2020 specification:
// https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_example_usage

#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace sycl;

using coeff_t = std::array<std::array<float, 3>, 3>;

struct coeff_struct_t {
  std::array<std::array<float, 3>, 3> c;
};

coeff_t get_coefficients() {
  return {{{1.0, 2.0, 3.0}, {1.1, 2.1, 3.1}, {1.2, 2.2, 3.2}}};
}

coeff_struct_t get_coefficient_struct() {
  return {{{{1.0, 2.0, 3.0}, {1.1, 2.1, 3.1}, {1.2, 2.2, 3.2}}}};
}

constexpr specialization_id<coeff_t> coeff_id;

constexpr specialization_id<coeff_struct_t> coeff_struct_id;

template <typename IN>
float calc_conv(const coeff_t &coeff, const IN &in, item<2> item_id) {
  float acc = 0;

  for (int i = -1; i <= 1; i++) {
    if (item_id[0] + i < 0 || item_id[0] + i >= in.get_range()[0])
      continue;
    for (int j = -1; j <= 1; j++) {
      if (item_id[1] + j < 0 || item_id[1] + j >= in.get_range()[1])
        continue;
      // The underlying JIT can see all the values of the array returned
      // by coeff.get().
      acc += coeff[i + 1][j + 1] * in[item_id[0] + i][item_id[1] + j];
    }
  }
  return acc;
}

template <typename KernelName, typename CP>
void do_conv(buffer<float, 2> in, buffer<float, 2> out, CP coeff_provider) {
  queue myQueue;

  myQueue.submit([&](handler &cgh) {
    auto in_acc = in.template get_access<access::mode::read>(cgh);
    auto out_acc = out.template get_access<access::mode::write>(cgh);

    // Set the coefficient of the convolution as constant.
    // This will build a specific kernel the coefficient available as literals.
    cgh.set_specialization_constant<coeff_id>(get_coefficients());
    cgh.set_specialization_constant<coeff_struct_id>(get_coefficient_struct());
    cgh.parallel_for<KernelName>(
        in.get_range(), [=](item<2> item_id, kernel_handler h) {
          auto coeff = coeff_provider(h);
          out_acc[item_id] = calc_conv(coeff, in_acc, item_id);
        });
  });

  myQueue.wait();
}

constexpr size_t N = 5;
constexpr size_t M = 4;

constexpr std::array<std::array<float, M>, N> expected = {
    {{17.1, 30.1, 43.0, 24.3},
     {41.3, 63.9, 82.8, 45.5},
     {72.5, 101.7, 120.6, 64.7},
     {103.7, 139.5, 158.4, 83.9},
     {77.7, 102.7, 115.0, 60.1}}};

template <typename Result, typename Expected>
void compare_result(const Result &result, const Expected &expected) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      if (std::abs(result[i][j] - expected[i][j]) > 0.1) {
        std::cout << "Wrong value " << result[i][j] << " on element " << i
                  << ", " << j << std::endl;
        exit(-1);
      }
    }
  }
}

int main() {

  buffer<float, 2> input{range<2>{N, M}};
  buffer<float, 2> output{range<2>{N, M}};

  // Launch an asynchronous kernel to initialize input
  queue myQueue;
  myQueue.submit([&](handler &cgh) {
    accessor input_acc{input, cgh, write_only};

    cgh.parallel_for(input.get_range(), [=](id<2> index) {
      input_acc[index] = index[0] * 2 + index[1];
    });
  });

  do_conv<class Convolution1>(input, output, [](kernel_handler &h) {
    return h.get_specialization_constant<coeff_id>();
  });

  compare_result(host_accessor{output, read_only}, expected);

  do_conv<class Convolution2>(input, output, [](kernel_handler &h) {
    return h.get_specialization_constant<coeff_struct_id>().c;
  });

  compare_result(host_accessor{output, read_only}, expected);

  std::cout << "Good computation!" << std::endl;
  return 0;
}
