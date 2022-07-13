// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- half_type.cpp - SYCL half type test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cmath>
#include <unordered_set>

using namespace cl::sycl;

constexpr size_t N = 100;

template <typename T> void assert_close(const T &C, const cl::sycl::half ref) {
  for (size_t i = 0; i < N; i++) {
    auto diff = C[i] - ref;
    assert(std::fabs(static_cast<float>(diff)) <
           std::numeric_limits<cl::sycl::half>::epsilon());
  }
}

void verify_add(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const half ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_add>(
        r, [=](id<1> index) { C[index] = A[index] + B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_min(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const half ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_min>(
        r, [=](id<1> index) { C[index] = A[index] - B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_mul(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const half ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_mul>(
        r, [=](id<1> index) { C[index] = A[index] * B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_div(queue &q, buffer<half, 1> &a, buffer<half, 1> &b, range<1> &r,
                const float ref) {
  buffer<half, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_div>(
        r, [=](id<1> index) { C[index] = A[index] / B[index]; });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_vec(queue &q) {
  half2 hvec(999);
  int a = 0;
  buffer<int, 1> e(&a, range<1>(1));
  q.submit([&](cl::sycl::handler &cgh) {
    auto E = e.get_access<access::mode::write>(cgh);
    cgh.single_task<class vec_of_half>([=]() {
      if (int(hvec.s0()) != 999 || int(hvec.s1()) != 999)
        E[0] = 1;
    });
  });
  assert(e.get_access<access::mode::read>()[0] == 0);
}

void verify_numeric_limits(queue &q) {
  // Verify on host side
  // Static member variables
  std::numeric_limits<cl::sycl::half>::is_specialized;
  std::numeric_limits<cl::sycl::half>::is_signed;
  std::numeric_limits<cl::sycl::half>::is_integer;
  std::numeric_limits<cl::sycl::half>::is_exact;
  std::numeric_limits<cl::sycl::half>::has_infinity;
  std::numeric_limits<cl::sycl::half>::has_quiet_NaN;
  std::numeric_limits<cl::sycl::half>::has_signaling_NaN;
  std::numeric_limits<cl::sycl::half>::has_denorm;
  std::numeric_limits<cl::sycl::half>::has_denorm_loss;
  std::numeric_limits<cl::sycl::half>::tinyness_before;
  std::numeric_limits<cl::sycl::half>::traps;
  std::numeric_limits<cl::sycl::half>::max_exponent10;
  std::numeric_limits<cl::sycl::half>::max_exponent;
  std::numeric_limits<cl::sycl::half>::min_exponent10;
  std::numeric_limits<cl::sycl::half>::min_exponent;
  std::numeric_limits<cl::sycl::half>::radix;
  std::numeric_limits<cl::sycl::half>::max_digits10;
  std::numeric_limits<cl::sycl::half>::digits;
  std::numeric_limits<cl::sycl::half>::is_bounded;
  std::numeric_limits<cl::sycl::half>::digits10;
  std::numeric_limits<cl::sycl::half>::is_modulo;
  std::numeric_limits<cl::sycl::half>::is_iec559;
  std::numeric_limits<cl::sycl::half>::round_style;

  // Static member functions
  std::numeric_limits<cl::sycl::half>::min();
  std::numeric_limits<cl::sycl::half>::max();
  std::numeric_limits<cl::sycl::half>::lowest();
  std::numeric_limits<cl::sycl::half>::epsilon();
  std::numeric_limits<cl::sycl::half>::round_error();
  std::numeric_limits<cl::sycl::half>::infinity();
  std::numeric_limits<cl::sycl::half>::quiet_NaN();
  std::numeric_limits<cl::sycl::half>::signaling_NaN();
  std::numeric_limits<cl::sycl::half>::denorm_min();

  // Verify in kernel function for device side check
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<class kernel>([]() {
      // Static member variables
      std::numeric_limits<cl::sycl::half>::is_specialized;
      std::numeric_limits<cl::sycl::half>::is_signed;
      std::numeric_limits<cl::sycl::half>::is_integer;
      std::numeric_limits<cl::sycl::half>::is_exact;
      std::numeric_limits<cl::sycl::half>::has_infinity;
      std::numeric_limits<cl::sycl::half>::has_quiet_NaN;
      std::numeric_limits<cl::sycl::half>::has_signaling_NaN;
      std::numeric_limits<cl::sycl::half>::has_denorm;
      std::numeric_limits<cl::sycl::half>::has_denorm_loss;
      std::numeric_limits<cl::sycl::half>::tinyness_before;
      std::numeric_limits<cl::sycl::half>::traps;
      std::numeric_limits<cl::sycl::half>::max_exponent10;
      std::numeric_limits<cl::sycl::half>::max_exponent;
      std::numeric_limits<cl::sycl::half>::min_exponent10;
      std::numeric_limits<cl::sycl::half>::min_exponent;
      std::numeric_limits<cl::sycl::half>::radix;
      std::numeric_limits<cl::sycl::half>::max_digits10;
      std::numeric_limits<cl::sycl::half>::digits;
      std::numeric_limits<cl::sycl::half>::is_bounded;
      std::numeric_limits<cl::sycl::half>::digits10;
      std::numeric_limits<cl::sycl::half>::is_modulo;
      std::numeric_limits<cl::sycl::half>::is_iec559;
      std::numeric_limits<cl::sycl::half>::round_style;

      // Static member functions
      std::numeric_limits<cl::sycl::half>::min();
      std::numeric_limits<cl::sycl::half>::max();
      std::numeric_limits<cl::sycl::half>::lowest();
      std::numeric_limits<cl::sycl::half>::epsilon();
      std::numeric_limits<cl::sycl::half>::round_error();
      std::numeric_limits<cl::sycl::half>::infinity();
      std::numeric_limits<cl::sycl::half>::quiet_NaN();
      std::numeric_limits<cl::sycl::half>::signaling_NaN();
      std::numeric_limits<cl::sycl::half>::denorm_min();
    });
  });
}

inline bool bitwise_comparison_fp16(const half val, const uint16_t exp) {
  return reinterpret_cast<const uint16_t &>(val) == exp;
}

inline bool bitwise_comparison_fp32(const half val, const uint32_t exp) {
  const float fp32 = static_cast<float>(val);
  return reinterpret_cast<const uint32_t &>(fp32) == exp;
}

constexpr void constexpr_verify_add() {
  constexpr half a{5.0}, b{2.0}, ref{7.0};
  constexpr half result = a + b;
  constexpr half diff = result - ref;
  constexpr auto sign = diff < 0 ? -1 : 1;
  static_assert(sign * static_cast<float>(diff) <
                    std::numeric_limits<cl::sycl::half>::epsilon(),
                "Constexpr add is wrong");
}

constexpr void constexpr_verify_sub() {
  constexpr half a{5.0f}, b{2.0}, ref{3.0};
  constexpr half result = a - b;
  constexpr half diff = result - ref;
  constexpr auto sign = diff < 0 ? -1 : 1;
  static_assert(sign * static_cast<float>(diff) <
                    std::numeric_limits<cl::sycl::half>::epsilon(),
                "Constexpr sub is wrong");
}

constexpr void constexpr_verify_mul() {
  constexpr half a{5.0f}, b{2.0}, ref{10.0};
  constexpr half result = a * b;
  constexpr half diff = result - ref;
  constexpr auto sign = diff < 0 ? -1 : 1;
  static_assert(sign * static_cast<float>(diff) <
                    std::numeric_limits<cl::sycl::half>::epsilon(),
                "Constexpr mul is wrong");
}

constexpr void constexpr_verify_div() {
  constexpr half a{5.0f}, b{2.0}, ref{2.5};
  constexpr half result = a / b;
  constexpr half diff = result - ref;
  constexpr auto sign = diff < 0 ? -1 : 1;
  static_assert(sign * static_cast<float>(diff) <
                    std::numeric_limits<cl::sycl::half>::epsilon(),
                "Constexpr div is wrong");
}

int main() {
  constexpr_verify_add();
  constexpr_verify_sub();
  constexpr_verify_mul();
  constexpr_verify_div();

  device dev{default_selector()};
  if (!dev.is_host() && !dev.has(sycl::aspect::fp16)) {
    std::cout << "This device doesn't support the extension cl_khr_fp16"
              << std::endl;
    return 0;
  }

  std::vector<half> vec_a(N, 5.0);
  std::vector<half> vec_b(N, 2.0);

  range<1> r(N);
  buffer<half, 1> a{vec_a.data(), r};
  buffer<half, 1> b{vec_b.data(), r};

  queue q{dev};

  verify_add(q, a, b, r, 7.0);
  verify_min(q, a, b, r, 3.0);
  verify_mul(q, a, b, r, 10.0);
  verify_div(q, a, b, r, 2.5);
  verify_vec(q);
  verify_numeric_limits(q);

  if (!dev.is_host()) {
    return 0;
  }

  // Basic tests: fp32->fp16
  // The following references are from `_cvtss_sh` with truncate mode.
  // +inf
  assert(bitwise_comparison_fp16(75514, 31744));
  // -inf
  assert(bitwise_comparison_fp16(-75514, 64512));
  // +0
  assert(bitwise_comparison_fp16(0.0, 0));
  // -0
  assert(bitwise_comparison_fp16(-0.0, 32768));
  // 1.9999f
  assert(bitwise_comparison_fp16(1.9999f, 0x4000));
  // nan
  assert(bitwise_comparison_fp16(0.0 / 0.0, 32256));
  assert(bitwise_comparison_fp16(-0.0 / 0.0, 32256));
  // special nan
  uint32_t special_nan = 0x7f800001;
  assert(
      bitwise_comparison_fp16(reinterpret_cast<float &>(special_nan), 32256));
  special_nan = 0xff800001;
  assert(
      bitwise_comparison_fp16(reinterpret_cast<float &>(special_nan), 65024));
  // subnormal
  assert(bitwise_comparison_fp16(9.8E-45, 0));
  assert(bitwise_comparison_fp16(-9.8E-45, 32768));
  uint32_t subnormal_in_16 = 0x38200000;
  // verify 0.000038146972 converts to 0.0000382
  assert(bitwise_comparison_fp16(reinterpret_cast<float &>(subnormal_in_16),
                                 0x0280));
  // overflow
  assert(bitwise_comparison_fp16(half(55504) * 3, 31744));
  assert(bitwise_comparison_fp16(half(-55504) * 3, 64512));
  // underflow
  assert(bitwise_comparison_fp16(half(8.1035e-05) / half(3), 453));

  // Basic tests: fp16->fp32
  // The following references are from `_cvtsh_ss`.
  // +inf
  const uint16_t pinf = 0x7a00;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(pinf),
                                 1195376640));
  // -inf
  const uint16_t ninf = 0xfa00;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(ninf),
                                 3342860288));
  // +0
  const uint16_t p0 = 0x0;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(p0), 0));
  // -0
  const uint16_t n0 = 0x8000;
  assert(
      bitwise_comparison_fp32(reinterpret_cast<const half &>(n0), 2147483648));
  // nan
  const uint16_t nan16 = 0x7a03;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(nan16),
                                 1195401216));
  // subnormal
  const uint16_t subnormal = 0x0005;
  assert(bitwise_comparison_fp32(reinterpret_cast<const half &>(subnormal),
                                 882900992));

  // std::hash<cl::sycl::half>
  std::unordered_set<half> sets;
  sets.insert(1.2);
  assert(sets.find(1.2) != sets.end());

  return 0;
}
