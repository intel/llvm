/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  math_vectorized.cpp
 *
 *  Description:
 *    math helpers for vectorized operations and fp16 operations
 **************************************************************************/

// REQUIRES: aspect-fp16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/math.hpp>

#include "../common.hpp"
#include "math_fixt.hpp"

template <typename BinaryOp, typename ValueT>
void vectorized_binary_kernel(ValueT *a, ValueT *b, unsigned *r,
                              bool need_relu) {
  unsigned ua = static_cast<unsigned>(*a);
  unsigned ub = static_cast<unsigned>(*b);
  *r = syclcompat::vectorized_binary<sycl::short2>(ua, ub, BinaryOp(),
                                                   need_relu);
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_binary(ValueT op1, ValueT op2, unsigned expected,
                            bool need_relu = false) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<ValueT, ValueT, unsigned>(grid, threads)
      .template launch_test<vectorized_binary_kernel<BinaryOp, ValueT>>(
          op1, op2, expected, need_relu);
}

template <typename UnaryOp, typename ValueT>
void vectorized_unary_kernel(ValueT *a, unsigned *r) {
  unsigned ua = static_cast<unsigned>(*a);
  *r = syclcompat::vectorized_unary<sycl::short2>(ua, UnaryOp());
}

template <typename UnaryOp, typename ValueT>
void test_vectorized_unary(ValueT op1, unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  UnaryOpTestLauncher<ValueT, unsigned>(grid, threads)
      .template launch_test<vectorized_unary_kernel<UnaryOp, ValueT>>(op1,
                                                                      expected);
}

template <typename ValueT>
void vectorized_sum_abs_diff_kernel(ValueT *a, ValueT *b, unsigned *r) {
  unsigned ua = static_cast<unsigned>(*a);
  unsigned ub = static_cast<unsigned>(*b);

  *r = syclcompat::vectorized_sum_abs_diff<sycl::short2>(ua, ub);
}

template <typename ValueT>
void test_vectorized_sum_abs_diff(ValueT op1, ValueT op2, unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<ValueT, ValueT, unsigned>(grid, threads)
      .template launch_test<vectorized_sum_abs_diff_kernel<ValueT>>(op1, op2,
                                                                    expected);
}

template <typename BinaryOp1, typename BinaryOp2, typename ValueT>
void vectorized_ternary_kernel(ValueT *a, ValueT *b, ValueT *c, unsigned *r,
                               bool need_relu) {
  unsigned ua = static_cast<unsigned>(*a);
  unsigned ub = static_cast<unsigned>(*b);
  unsigned uc = static_cast<unsigned>(*c);
  *r = syclcompat::vectorized_ternary<sycl::short2>(ua, ub, uc, BinaryOp1(),
                                                    BinaryOp2(), need_relu);
}

template <typename BinaryOp1, typename BinaryOp2, typename ValueT>
void test_vectorized_ternary(ValueT op1, ValueT op2, ValueT op3,
                             unsigned expected, bool need_relu = false) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  TernaryOpTestLauncher<ValueT, ValueT, unsigned>(grid, threads)
      .template launch_test<
          vectorized_ternary_kernel<BinaryOp1, BinaryOp2, ValueT>>(
          op1, op2, op3, expected, need_relu);
}

template <typename BinaryOp, typename ValueT>
void vectorized_with_pred_kernel(ValueT *a, ValueT *b, unsigned *r,
                                 bool *pred_hi, bool *pred_lo) {
  unsigned ua = static_cast<unsigned>(*a);
  unsigned ub = static_cast<unsigned>(*b);

  *r = syclcompat::vectorized_with_pred<short>(ua, ub, BinaryOp(), pred_hi,
                                               pred_lo);
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_with_pred(ValueT op1, ValueT op2, unsigned expected,
                               bool expected_hi, bool expected_lo) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<ValueT, ValueT, unsigned>(grid, threads)
      .template launch_test<vectorized_with_pred_kernel<BinaryOp, ValueT>>(
          op1, op2, expected, expected_hi, expected_lo);
}

int main() {
  test_vectorized_binary<syclcompat::abs_diff, uint32_t>(0x00010002, 0x00040002,
                                                         0x00030000);
  test_vectorized_binary<syclcompat::add_sat, uint32_t>(0x00020002, 0xFFFDFFFF,
                                                        0xFFFF0001);
  test_vectorized_binary<syclcompat::rhadd, uint32_t>(0x00010008, 0x00020001,
                                                      0x00020005);
  test_vectorized_binary<syclcompat::hadd, uint32_t>(0x00010003, 0x00020005,
                                                     0x00010004);
  test_vectorized_binary<syclcompat::maximum, uint32_t>(0x0FFF0000, 0x00000FFF,
                                                        0x0FFF0FFF);
  test_vectorized_binary<syclcompat::minimum, uint32_t>(0x0FFF0000, 0x00000FFF,
                                                        0x00000000);
  test_vectorized_binary<syclcompat::sub_sat, uint32_t>(0xFFFB0005, 0x00030008,
                                                        0xFFF8FFFD);
  test_vectorized_binary<syclcompat::abs_diff, uint32_t>(0x00010002, 0x00040002,
                                                         0x00030000, true);
  test_vectorized_binary<syclcompat::add_sat, uint32_t>(0x00020002, 0xFFFDFFFF,
                                                        0x00000001, true);
  test_vectorized_binary<syclcompat::rhadd, uint32_t>(0x00010008, 0x00020001,
                                                      0x00020005, true);
  test_vectorized_binary<syclcompat::hadd, uint32_t>(0x00010003, 0x00020005,
                                                     0x00010004, true);
  test_vectorized_binary<syclcompat::maximum, uint32_t>(0x0FFF0000, 0x00000FFF,
                                                        0x0FFF0FFF, true);
  test_vectorized_binary<syclcompat::minimum, uint32_t>(0x0FFF0000, 0x00000FFF,
                                                        0x00000000, true);
  test_vectorized_binary<syclcompat::sub_sat, uint32_t>(0xFFFB0005, 0x00030008,
                                                        0x00000000, true);
  test_vectorized_unary<syclcompat::abs, uint32_t>(0xFFFBFFFD, 0x00050003);
  test_vectorized_sum_abs_diff<uint32_t>(0x00010002, 0x00040002, 0x00000003);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004, true);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00050004);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00050004, true);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004, true);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00010002);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum, uint32_t>(
      0x00010002, 0x00040002, 0x00080004, 0x00010002, true);
  test_vectorized_with_pred<syclcompat::maximum, uint32_t>(
      0x00010002, 0x00040002, 0x00040002, false, true);
  test_vectorized_with_pred<syclcompat::minimum, uint32_t>(
      0x00010002, 0x00040002, 0x00010002, true, true);

  return 0;
}
