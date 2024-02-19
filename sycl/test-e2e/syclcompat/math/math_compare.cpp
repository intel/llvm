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
 *  math_compare.cpp
 *
 *  Description:
 *    math helpers tests
 **************************************************************************/

// The original source was under the license below:
// ===------------------- math.cpp ---------- -*- C++ -* ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// REQUIRES: aspect-fp16

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/half_type.hpp>
#include <syclcompat/math.hpp>

#include "../common.hpp"
#include "math_fixt.hpp"

template <typename ValueT>
void compare_equal_kernel(ValueT *a, ValueT *b, bool *r) {
  *r = syclcompat::compare(*a, *b, std::equal_to<>());
}

template <typename ValueT>
void compare_not_equal_kernel(ValueT *a, ValueT *b, bool *r) {
  *r = syclcompat::compare(*a, *b, std::not_equal_to<>());
}

template <typename ValueT> void test_compare() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(1.0);
  ValueT op2 = sycl::nan(static_cast<unsigned int>(0));

  //  1.0 == 1.0 -> True
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_equal_kernel<ValueT>>(op1, op1, true);
  //  NaN == 1.0 -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_equal_kernel<ValueT>>(op2, op1, false);
  //  1.0 == NaN -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_equal_kernel<ValueT>>(op1, op2, false);
  //  NaN == NaN -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_equal_kernel<ValueT>>(op2, op2, false);

  //  1.0 != 1.0 -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_not_equal_kernel<ValueT>>(op1, op1, false);
  //  NaN != 1.0 -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_not_equal_kernel<ValueT>>(op2, op1, false);
  //  1.0 != NaN -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_not_equal_kernel<ValueT>>(op1, op2, false);
  //  NaN != NaN -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<compare_not_equal_kernel<ValueT>>(op2, op2, false);
}

template <typename ValueT>
void unordered_compare_equal_kernel(ValueT *a, ValueT *b, bool *r) {
  *r = syclcompat::unordered_compare(*a, *b, std::equal_to<>());
}

template <typename ValueT>
void unordered_compare_not_equal_kernel(ValueT *a, ValueT *b, bool *r) {
  *r = syclcompat::unordered_compare(*a, *b, std::not_equal_to<>());
}

template <typename ValueT, typename ValueU = ValueT>
void test_unordered_compare() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(1.0);
  ValueT op2 = sycl::nan(static_cast<unsigned int>(0));

  // Unordered comparison checks if either operand is NaN, or the binaryop holds
  // true
  //  1.0 == 1.0 -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<unordered_compare_equal_kernel<ValueT>>(op1, op1,
                                                                    true);
  //  NaN == 1.0 -> True
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<unordered_compare_equal_kernel<ValueT>>(op2, op1,
                                                                    true);
  //  1.0 == NaN -> True
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<unordered_compare_equal_kernel<ValueT>>(op1, op2,
                                                                    true);
  //  NaN == NaN -> True
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<unordered_compare_equal_kernel<ValueT>>(op2, op2,
                                                                    true);
  //  1.0 != 1.0 -> False
  BinaryOpTestLauncher<ValueT, ValueT, bool>(grid, threads)
      .template launch_test<unordered_compare_not_equal_kernel<ValueT>>(
          op1, op1, false);
  // No need to check again if either operand is NaN
}

void isnan_kernel(sycl::float2 *a, sycl::float2 *r) {
  *r = syclcompat::isnan(*a);
}

void test_isnan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  sycl::float2 op1 = {sycl::nan(static_cast<unsigned int>(0)), 1.0f};
  // bool2 does not exist,1.0 and 0.0 floats are used for true
  // and false instead.
  sycl::float2 expect = {1.0, 0.0};

  UnaryOpTestLauncher<sycl::float2>(grid, threads)
      .template launch_test<isnan_kernel>(op1, expect);
}

template <class F> void test_vectorized_binary() {
  unsigned u;
  syclcompat::vectorized_binary<sycl::short2>(u, u, F());
}

void test_vectorized_binary_abs_diff() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::abs_diff;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_add_sat() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::add_sat;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_rhadd() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::rhadd;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_hadd() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::hadd;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_maximum() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::maximum;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_minimum() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::minimum;
  test_vectorized_binary<F>();
}

void test_vectorized_binary_sub_sat() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using F = syclcompat::sub_sat;
  test_vectorized_binary<F>();
}

void test_vectorized_unary() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  unsigned u;
  using F = syclcompat::abs;
  syclcompat::vectorized_unary<sycl::short2>(u, F());
}

void test_vectorized_sum_abs_diff() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  unsigned u;
  syclcompat::vectorized_sum_abs_diff<sycl::short2>(u, u);
}

int main() {
  INSTANTIATE_ALL_TYPES(floating_type_list, test_compare);
  INSTANTIATE_ALL_TYPES(floating_type_list, test_unordered_compare);
  INSTANTIATE_ALL_TYPES(floating_type_list, test_compare_both);
  INSTANTIATE_ALL_TYPES(floating_type_list, test_unordered_compare_both);
  test_isnan();

  // TODO: These currently only check API
  // test_compare_both();
  // test_unordered_compare_both();
  test_vectorized_binary_abs_diff();
  test_vectorized_binary_add_sat();
  test_vectorized_binary_rhadd();
  test_vectorized_binary_hadd();
  test_vectorized_binary_maximum();
  test_vectorized_binary_minimum();
  test_vectorized_binary_sub_sat();
  test_vectorized_unary();
  test_vectorized_sum_abs_diff();

  return 0;
}
