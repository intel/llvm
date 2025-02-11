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
void vectorized_binary_kernel(unsigned *a, unsigned *b, unsigned *r,
                              bool need_relu) {
  *r = syclcompat::vectorized_binary<ValueT>(*a, *b, BinaryOp(), need_relu);
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_binary(unsigned op1, unsigned op2, unsigned expected,
                            bool need_relu = false) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<unsigned, unsigned, unsigned>(grid, threads)
      .template launch_test<vectorized_binary_kernel<BinaryOp, ValueT>>(
          op1, op2, expected, need_relu);
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_binary_logical(unsigned op1, unsigned op2,
                                    unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<unsigned, unsigned, unsigned>(grid, threads)
      .template launch_test<vectorized_binary_kernel<BinaryOp, ValueT>>(
          op1, op2, expected, false);
}

template <typename UnaryOp, typename ValueT>
void vectorized_unary_kernel(unsigned *a, unsigned *r) {
  *r = syclcompat::vectorized_unary<ValueT>(*a, UnaryOp());
}

template <typename UnaryOp, typename ValueT>
void test_vectorized_unary(unsigned op1, unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  UnaryOpTestLauncher<unsigned, unsigned>(grid, threads)
      .template launch_test<vectorized_unary_kernel<UnaryOp, ValueT>>(op1,
                                                                      expected);
}

template <typename ValueT>
void vectorized_sum_abs_diff_kernel(unsigned *a, unsigned *b, unsigned *r) {
  *r = syclcompat::vectorized_sum_abs_diff<ValueT>(*a, *b);
}

template <typename ValueT>
void test_vectorized_sum_abs_diff(unsigned op1, unsigned op2,
                                  unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<unsigned, unsigned, unsigned>(grid, threads)
      .template launch_test<vectorized_sum_abs_diff_kernel<ValueT>>(op1, op2,
                                                                    expected);
}

template <typename BinaryOp1, typename BinaryOp2, typename ValueT>
void vectorized_ternary_kernel(unsigned *a, unsigned *b, unsigned *c,
                               unsigned *r, bool need_relu) {
  *r = syclcompat::vectorized_ternary<ValueT>(*a, *b, *c, BinaryOp1(),
                                              BinaryOp2(), need_relu);
}

template <typename BinaryOp1, typename BinaryOp2, typename ValueT>
void test_vectorized_ternary(unsigned op1, unsigned op2, unsigned op3,
                             unsigned expected, bool need_relu = false) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  TernaryOpTestLauncher<unsigned, unsigned, unsigned>(grid, threads)
      .template launch_test<
          vectorized_ternary_kernel<BinaryOp1, BinaryOp2, ValueT>>(
          op1, op2, op3, expected, need_relu);
}

template <typename BinaryOp, typename ValueT>
void vectorized_binary_with_pred_kernel(unsigned *a, unsigned *b, unsigned *r,
                                        bool *pred_hi, bool *pred_lo) {
  *r = syclcompat::vectorized_binary_with_pred<ValueT>(*a, *b, BinaryOp(),
                                                       pred_hi, pred_lo);
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_binary_with_pred(unsigned op1, unsigned op2,
                                      unsigned expected, bool pred_hi,
                                      bool pred_lo) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<unsigned, unsigned, unsigned>(grid, threads)
      .template launch_test<
          vectorized_binary_with_pred_kernel<BinaryOp, ValueT>>(
          op1, op2, expected, pred_hi, pred_lo);
}

int main() {
  test_vectorized_binary<syclcompat::abs_diff, sycl::short2>(
      0x00010002, 0x00040002, 0x00030000);
  test_vectorized_binary<syclcompat::add_sat, sycl::short2>(
      0x00020002, 0xFFFDFFFF, 0xFFFF0001);
  test_vectorized_binary<syclcompat::rhadd, sycl::short2>(
      0x00010008, 0x00020001, 0x00020005);
  test_vectorized_binary<syclcompat::hadd, sycl::short2>(0x00010003, 0x00020005,
                                                         0x00010004);
  test_vectorized_binary<syclcompat::maximum, sycl::short2>(
      0x0FFF0000, 0x00000FFF, 0x0FFF0FFF);
  test_vectorized_binary<syclcompat::minimum, sycl::short2>(
      0x0FFF0000, 0x00000FFF, 0x00000000);
  test_vectorized_binary<syclcompat::sub_sat, sycl::short2>(
      0xFFFB0005, 0x00030008, 0xFFF8FFFD);
  test_vectorized_binary<syclcompat::abs_diff, sycl::short2>(
      0x00010002, 0x00040002, 0x00030000, true);
  test_vectorized_binary<syclcompat::add_sat, sycl::short2>(
      0x00020002, 0xFFFDFFFF, 0x00000001, true);
  test_vectorized_binary<syclcompat::rhadd, sycl::short2>(
      0x00010008, 0x00020001, 0x00020005, true);
  test_vectorized_binary<syclcompat::hadd, sycl::short2>(0x00010003, 0x00020005,
                                                         0x00010004, true);
  test_vectorized_binary<syclcompat::maximum, sycl::short2>(
      0x0FFF0000, 0x00000FFF, 0x0FFF0FFF, true);
  test_vectorized_binary<syclcompat::minimum, sycl::short2>(
      0x0FFF0000, 0x00000FFF, 0x00000000, true);
  test_vectorized_binary<syclcompat::sub_sat, sycl::short2>(
      0xFFFB0005, 0x00030008, 0x00000000, true);
  test_vectorized_unary<syclcompat::abs, sycl::short2>(0xFFFBFFFD, 0x00050003);
  test_vectorized_sum_abs_diff<sycl::ushort2>(0x00010002, 0x00040002,
                                              0x00000003);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, sycl::ushort2>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, sycl::ushort2>(
      0x00010002, 0x00040002, 0x00080004, 0x00080004, true);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, sycl::ushort2>(
      0x00010002, 0x00040002, 0x00080004, 0x00050004);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, sycl::ushort2>(
      0x00010002, 0x00040002, 0x00080004, 0x00050004, true);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum,
                          sycl::ushort2>(0x00010002, 0x00040002, 0x00080004,
                                         0x00080004);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum,
                          sycl::ushort2>(0x00010002, 0x00040002, 0x00080004,
                                         0x00080004, true);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum,
                          sycl::ushort2>(0x00010002, 0x00040002, 0x00080004,
                                         0x00010002);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum,
                          sycl::ushort2>(0x00010002, 0x00040002, 0x00080004,
                                         0x00010002, true);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, sycl::short2>(
      0x80010002, 0x00040002, 0x00080004, 0x00080004);
  test_vectorized_ternary<std::plus<>, syclcompat::maximum, sycl::short2>(
      0x80010002, 0x00040002, 0x00080004, 0x00080004, true);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, sycl::short2>(
      0x80010002, 0x00040002, 0x00080004, 0x80050004);
  test_vectorized_ternary<std::plus<>, syclcompat::minimum, sycl::short2>(
      0x80010002, 0x00040002, 0x00080004, 0x00000004, true);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum,
                          sycl::short2>(0x80010002, 0x00040002, 0x00080004,
                                        0x00080004);
  test_vectorized_ternary<syclcompat::maximum, syclcompat::maximum,
                          sycl::short2>(0x80010002, 0x00040002, 0x00080004,
                                        0x00080004, true);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum,
                          sycl::short2>(0x80010002, 0x00040002, 0x00080004,
                                        0x80010002);
  test_vectorized_ternary<syclcompat::minimum, syclcompat::minimum,
                          sycl::short2>(0x80010002, 0x00040002, 0x00080004,
                                        0x00000002, true);
  test_vectorized_binary_with_pred<syclcompat::maximum, sycl::short2>(
      0x80010002, 0x00040002, 0x00040002, false, true);
  test_vectorized_binary_with_pred<syclcompat::minimum, sycl::short2>(
      0x80010002, 0x00040002, 0x80010002, true, true);
  test_vectorized_binary_with_pred<syclcompat::maximum, sycl::ushort2>(
      0x80010002, 0x00040002, 0x80010002, true, true);
  test_vectorized_binary_with_pred<syclcompat::minimum, sycl::ushort2>(
      0x80010002, 0x00040002, 0x00040002, false, true);

  // Logical Binary Operators v2
  test_vectorized_binary_logical<std::equal_to<>, sycl::short2>(
      0xFFF00002, 0xFFF00001, 0xFFFF0000);
  test_vectorized_binary_logical<std::equal_to<>, sycl::short2>(
      0x0001F00F, 0x0003F00F, 0x0000FFFF);

  test_vectorized_binary_logical<std::not_equal_to<>, sycl::short2>(
      0xFFF00002, 0xFFF00001, 0x0000FFFF);
  test_vectorized_binary_logical<std::not_equal_to<>, sycl::short2>(
      0x0001F00F, 0x0003F00F, 0xFFFF0000);

  test_vectorized_binary_logical<std::greater_equal<>, sycl::short2>(
      0xFFF00002, 0xFFF00001, 0xFFFFFFFF);
  test_vectorized_binary_logical<std::greater_equal<>, sycl::short2>(
      0x0001F00F, 0x0003F001, 0x0000FFFF);

  test_vectorized_binary_logical<std::greater<>, sycl::short2>(
      0xFFF00002, 0xFFF00001, 0x0000FFFF);
  test_vectorized_binary_logical<std::greater<>, sycl::short2>(
      0x0003F00F, 0x0001F00F, 0xFFFF0000);

  test_vectorized_binary_logical<std::less_equal<>, sycl::short2>(
      0xFFF00001, 0xF0F00002, 0x0000FFFF);
  test_vectorized_binary_logical<std::less_equal<>, sycl::short2>(
      0x0001FF0F, 0x0003F00F, 0xFFFF0000);

  test_vectorized_binary_logical<std::less<>, sycl::short2>(
      0xFFF00001, 0xFFF00002, 0x0000FFFF);
  test_vectorized_binary_logical<std::less<>, sycl::short2>(
      0x0001F00F, 0x0003F00F, 0xFFFF0000);

  // Logical Binary Operators v4
  test_vectorized_binary_logical<std::equal_to<>, sycl::uchar4>(
      0x0001F00F, 0x0003F00F, 0xFF00FFFF);
  test_vectorized_binary_logical<std::equal_to<>, sycl::uchar4>(
      0x0102F0F0, 0x0202F0FF, 0x00FFFF00);

  test_vectorized_binary_logical<std::not_equal_to<>, sycl::uchar4>(
      0x0001F00F, 0xFF01F10F, 0xFF00FF00);
  test_vectorized_binary_logical<std::not_equal_to<>, sycl::uchar4>(
      0x0201F0F0, 0x0202F0FF, 0x00FF00FF);

  test_vectorized_binary_logical<std::greater_equal<>, sycl::uchar4>(
      0xFFF00002, 0xFFF10101, 0xFF0000FF);
  test_vectorized_binary_logical<std::greater_equal<>, sycl::uchar4>(
      0x0001F1F0, 0x0103F001, 0x0000FFFF);

  test_vectorized_binary_logical<std::greater<>, sycl::uchar4>(
      0xFFF00002, 0xF0F00001, 0xFF0000FF);
  test_vectorized_binary_logical<std::greater<>, sycl::uchar4>(
      0x0103F0F1, 0x0102F0F0, 0x00FF00FF);

  test_vectorized_binary_logical<std::less_equal<>, sycl::uchar4>(
      0xFFF10001, 0xFFF00100, 0xFF00FF00);
  test_vectorized_binary_logical<std::less_equal<>, sycl::uchar4>(
      0x0101F1F0, 0x0003F0F1, 0x00FF00FF);

  test_vectorized_binary_logical<std::less<>, sycl::uchar4>(
      0xFFF10001, 0xFFF20100, 0x00FFFF00);
  test_vectorized_binary_logical<std::less<>, sycl::uchar4>(
      0x0101F1F0, 0x0102F1F1, 0x00FF00FF);

  return 0;
}
