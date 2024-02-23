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
 *  math_basic.cpp
 *
 *  Description:
 *    basic math helper functions tests
 **************************************************************************/

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/dims.hpp>
#include <syclcompat/math.hpp>

#include "../common.hpp"
#include "math_fixt.hpp"

template <typename ValueT, typename ValueU>
inline void max_kernel(ValueT *a, ValueU *b,
                       std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::max(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_max() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(5);
  constexpr ValueU op2 = static_cast<ValueU>(10);
  constexpr std::common_type_t<ValueT, ValueU> res = static_cast<ValueU>(10);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<max_kernel<ValueT, ValueU>>(op1, op2, res);
}

template <typename ValueT, typename ValueU>
inline void min_kernel(ValueT *a, ValueU *b,
                       std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::min(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_min() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(5);
  constexpr ValueU op2 = static_cast<ValueU>(10);
  constexpr std::common_type_t<ValueT, ValueU> res =
      static_cast<std::common_type_t<ValueT, ValueU>>(5);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<min_kernel<ValueT, ValueU>>(op1, op2, res);
}

template <typename ValueT, typename ValueU>
inline void pow_kernel(ValueT *a, ValueU *b, ValueT *r) {
  *r = syclcompat::pow(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_pow() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  // 3 ** 3 = 27
  constexpr ValueT op1 = static_cast<ValueT>(3);
  constexpr ValueU op2 = static_cast<ValueU>(3);
  constexpr ValueT res = static_cast<ValueT>(27);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<pow_kernel<ValueT, ValueU>>(op1, op2, res);
}

template <typename ValueT> inline void relu_kernel(ValueT *a, ValueT *r) {
  *r = syclcompat::relu(*a);
}

template <typename ValueT> void test_syclcompat_relu() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  // relu(3) = 3, relu(-value) = 0
  constexpr ValueT op1 = static_cast<ValueT>(3);
  constexpr ValueT res1 = static_cast<ValueT>(3);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<relu_kernel<ValueT>>(op1, res1);

  constexpr ValueT op2 = static_cast<ValueT>(-3);
  constexpr ValueT res2 = static_cast<ValueT>(0);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<relu_kernel<ValueT>>(op2, res2);

  using ValueU = sycl::vec<ValueT, 2>;
  constexpr ValueU op3{op1, op2};
  constexpr ValueU res3{res1, res2};
  UnaryOpTestLauncher<ValueU>(grid, threads)
      .template launch_test<relu_kernel<ValueU>>(op3, res3);

  using ValueV = sycl::marray<ValueT, 2>;
  constexpr ValueV op4{op1, op2};
  constexpr ValueV res4{res1, res2};
  UnaryOpTestLauncher<ValueV>(grid, threads)
      .template launch_test<relu_kernel<ValueV>>(op4, res4);
}

template <typename ValueT> inline void cbrt_kernel(ValueT *a, ValueT *r) {
  *r = syclcompat::cbrt(*a);
}

template <typename ValueT> void test_syclcompat_cbrt() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  constexpr ValueT op1 = static_cast<ValueT>(1);
  constexpr ValueT res1 = static_cast<ValueT>(1);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<cbrt_kernel<ValueT>>(op1, res1);

  constexpr ValueT op2 = static_cast<ValueT>(64);
  constexpr ValueT res2 = static_cast<ValueT>(4);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<cbrt_kernel<ValueT>>(op2, res2);
}

int main() {
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_max);
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_min);

  // Basic testing of deduction to avoid combinatorial explosion
  test_syclcompat_max<double, float>();
  test_syclcompat_max<long, int>();
  test_syclcompat_min<double, float>();
  test_syclcompat_min<long, int>();

  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_max);
  test_syclcompat_pow<float, int>();
  test_syclcompat_pow<double, int>();

  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_relu);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_cbrt);

  return 0;
}
