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
void vectorized_binary_kernel(ValueT *a, ValueT *b, unsigned *r) {
  unsigned ua = static_cast<unsigned>(*a);
  unsigned ub = static_cast<unsigned>(*b);
  *r = syclcompat::vectorized_binary<sycl::short2>(ua, ub, BinaryOp());
}

template <typename BinaryOp, typename ValueT>
void test_vectorized_binary(ValueT op1, ValueT op2, unsigned expected) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  BinaryOpTestLauncher<ValueT, ValueT, unsigned>(grid, threads)
      .template launch_test<vectorized_binary_kernel<BinaryOp, ValueT>>(
          op1, op2, expected);
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
  test_vectorized_unary<syclcompat::abs, uint32_t>(0xFFFBFFFD, 0x00050003);
  test_vectorized_sum_abs_diff<uint32_t>(0x00010002, 0x00040002, 0x00000003);

  return 0;
}
