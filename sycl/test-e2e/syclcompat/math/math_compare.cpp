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

// RUN: %{build} -o %t.out
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

template <typename Container>
void compare_equal_vec_kernel(Container *a, Container *b, Container *r) {
  *r = syclcompat::compare(*a, *b, std::equal_to<>());
}

template <typename Container>
void compare_not_equal_vec_kernel(Container *a, Container *b, Container *r) {
  *r = syclcompat::compare(*a, *b, std::not_equal_to<>());
}

template <typename ValueT> void test_compare_vec() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  Container op2 = {static_cast<ValueT>(1.0),
                   sycl::nan(static_cast<unsigned int>(0))};

  // bool2 does not exist, 1.0 and 0.0 floats are used for true
  // and false instead.
  //  1.0 == 1.0, 2.0 == NaN -> {true, false}
  constexpr Container res1 = {1.0, 0.0};
  BinaryOpTestLauncher<Container, Container>(grid, threads)
      .template launch_test<compare_equal_vec_kernel<Container>>(op1, op2,
                                                                 res1);
  //  1.0 != 1.0, 2.0 != NaN -> {false, false}
  constexpr Container res2 = {0.0, 0.0};
  BinaryOpTestLauncher<Container, Container>(grid, threads)
      .template launch_test<compare_not_equal_vec_kernel<Container>>(op1, op2,
                                                                     res2);
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
  //  1.0 == 1.0 -> True
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

template <typename Container>
void unordered_compare_equal_vec_kernel(Container *a, Container *b,
                                        Container *r) {
  *r = syclcompat::unordered_compare(*a, *b, std::equal_to<>());
}

template <typename Container>
void unordered_compare_not_equal_vec_kernel(Container *a, Container *b,
                                            Container *r) {
  *r = syclcompat::unordered_compare(*a, *b, std::not_equal_to<>());
}

template <typename ValueT> void test_unordered_compare_vec() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  Container op2 = {static_cast<ValueT>(1.0),
                   sycl::nan(static_cast<unsigned int>(0))};

  // bool2 does not exist, 1.0 and 0.0 floats are used for true
  // and false instead.
  //  1.0 == 1.0, 2.0 == NaN -> {true, true}
  constexpr Container res1 = {1.0, 1.0};
  BinaryOpTestLauncher<Container, Container>(grid, threads)
      .template launch_test<unordered_compare_equal_vec_kernel<Container>>(
          op1, op2, res1);
  //  1.0 != 1.0, 2.0 != NaN -> {false, true}
  constexpr Container res2 = {0.0, 1.0};
  BinaryOpTestLauncher<Container, Container>(grid, threads)
      .template launch_test<unordered_compare_not_equal_vec_kernel<Container>>(
          op1, op2, res2);
}

template <typename Container>
void compare_both_kernel(Container *a, Container *b, bool *r) {
  *r = syclcompat::compare_both(*a, *b, std::equal_to<>());
}

template <typename ValueT> void test_compare_both() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  Container op2 = {static_cast<ValueT>(1.0),
                   sycl::nan(static_cast<unsigned int>(0))};

  //  1.0 == 1.0, 2.0 == NaN -> {true, false} -> false
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<compare_both_kernel<Container>>(op1, op2, false);

  //  1.0 == 1.0, 2.0 == 2.0 -> {true, true} -> true
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<compare_both_kernel<Container>>(op1, op1, true);

  //  1.0 == 1.0, NaN == NaN -> {true, false} -> false
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<compare_both_kernel<Container>>(op2, op2, false);
}

template <typename Container>
void unordered_compare_both_kernel(Container *a, Container *b, bool *r) {
  *r = syclcompat::unordered_compare_both(*a, *b, std::equal_to<>());
}

template <typename ValueT> void test_unordered_compare_both() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  Container op2 = {static_cast<ValueT>(1.0),
                   sycl::nan(static_cast<unsigned int>(0))};

  //  1.0 == 1.0, 2.0 == NaN -> {true, true} -> true
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<unordered_compare_both_kernel<Container>>(op1, op2,
                                                                      true);
  //  1.0 == 1.0, 2.0 == 2.0 -> {true, true} -> true
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<unordered_compare_both_kernel<Container>>(op1, op1,
                                                                      true);
  //  1.0 == 1.0, NaN == NaN -> {true, true} -> true
  BinaryOpTestLauncher<Container, Container, bool>(grid, threads)
      .template launch_test<unordered_compare_both_kernel<Container>>(op2, op2,
                                                                      true);
}

template <typename Container>
void compare_mask_kernel(Container *a, Container *b, unsigned *r) {
  *r = syclcompat::compare_mask(*a, *b, std::equal_to<>());
}

template <typename ValueT> void test_compare_mask() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  constexpr Container op2 = {static_cast<ValueT>(2.0),
                             static_cast<ValueT>(1.0)};
  constexpr Container op3 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(3.0)};
  constexpr Container op4 = {static_cast<ValueT>(3.0),
                             static_cast<ValueT>(2.0)};
  Container op5 = {sycl::nan(static_cast<unsigned int>(0)),
                   sycl::nan(static_cast<unsigned int>(0))};

  //  1.0 == 1.0, 2.0 == 2.0 -> 0xffffffff
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<compare_mask_kernel<Container>>(op1, op1,
                                                            0xffffffff);

  //  1.0 == 2.0, 2.0 == 1.0 -> 0x00000000
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<compare_mask_kernel<Container>>(op1, op2,
                                                            0x00000000);

  //  1.0 == 1.0, 2.0 == 3.0 -> 0xffff0000
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<compare_mask_kernel<Container>>(op1, op3,
                                                            0xffff0000);

  //  1.0 == 3.0, 2.0 == 2.0 -> 0x0000ffff
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<compare_mask_kernel<Container>>(op1, op4,
                                                            0x0000ffff);

  //  1.0 == NaN, 2.0 == NaN -> 0x00000000
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<compare_mask_kernel<Container>>(op1, op5,
                                                            0x00000000);
}

template <typename Container>
void unordered_compare_mask_kernel(Container *a, Container *b, unsigned *r) {
  *r = syclcompat::unordered_compare_mask(*a, *b, std::equal_to<>());
}

template <typename ValueT> void test_unordered_compare_mask() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  using Container = sycl::vec<ValueT, 2>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr Container op1 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(2.0)};
  constexpr Container op2 = {static_cast<ValueT>(2.0),
                             static_cast<ValueT>(1.0)};
  constexpr Container op3 = {static_cast<ValueT>(1.0),
                             static_cast<ValueT>(3.0)};
  constexpr Container op4 = {static_cast<ValueT>(3.0),
                             static_cast<ValueT>(2.0)};
  Container op5 = {sycl::nan(static_cast<unsigned int>(0)),
                   sycl::nan(static_cast<unsigned int>(0))};

  //  1.0 == 1.0, 2.0 == 2.0 -> 0xffffffff
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<unordered_compare_mask_kernel<Container>>(
          op1, op1, 0xffffffff);

  //  1.0 == 2.0, 2.0 == 1.0 -> 0x00000000
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<unordered_compare_mask_kernel<Container>>(
          op1, op2, 0x00000000);

  //  1.0 == 1.0, 2.0 == 3.0 -> 0xffff0000
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<unordered_compare_mask_kernel<Container>>(
          op1, op3, 0xffff0000);

  //  1.0 == 3.0, 2.0 == 2.0 -> 0x0000ffff
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<unordered_compare_mask_kernel<Container>>(
          op1, op4, 0x0000ffff);

  //  1.0 == NaN, 2.0 == NaN -> 0xffffffff
  BinaryOpTestLauncher<Container, Container, unsigned>(grid, threads)
      .template launch_test<unordered_compare_mask_kernel<Container>>(
          op1, op5, 0xffffffff);
}

int main() {
  INSTANTIATE_ALL_TYPES(fp_type_list, test_compare);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_unordered_compare);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_compare_vec);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_unordered_compare_vec);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_compare_both);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_unordered_compare_both);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_compare_mask);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_unordered_compare_mask);

  return 0;
}
