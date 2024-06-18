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
 *  math_ops.cpp
 *
 *  Description:
 *    tests for non-vectorized math helper functions
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
inline void fmin_nan_kernel(ValueT *a, ValueU *b,
                            std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::fmin_nan(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_fmin_nan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(5);
  constexpr ValueU op2 = static_cast<ValueU>(10);
  ValueU op3 = sycl::nan(static_cast<unsigned int>(0));

  constexpr std::common_type_t<ValueT, ValueU> res =
      static_cast<std::common_type_t<ValueT, ValueU>>(5);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ValueT, ValueU>>(op1, op2, res);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ValueT, ValueU>>(op1, op3, op3);
}

template <typename ValueT, typename ValueU>
inline void fmax_nan_kernel(ValueT *a, ValueU *b,
                            std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::fmax_nan(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_fmax_nan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  constexpr ValueT op1 = static_cast<ValueT>(5);
  constexpr ValueU op2 = static_cast<ValueU>(10);
  ValueU op3 = sycl::nan(static_cast<unsigned int>(0));

  constexpr std::common_type_t<ValueT, ValueU> res =
      static_cast<std::common_type_t<ValueT, ValueU>>(10);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ValueT, ValueU>>(op1, op2, res);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ValueT, ValueU>>(op1, op3, op3);
}

template <typename ValueT, typename ValueU>
inline void pow_kernel(ValueT *a, ValueU *b, ValueT *r) {
  *r = syclcompat::pow(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_pow() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  // FIXME: non-floating point values default to double, requires fp64. Change
  // when problem is solver at the header.
  if constexpr (!std::is_floating_point_v<ValueT>) {
    if (!syclcompat::get_current_device().has(sycl::aspect::fp64)) {
      std::cout << "  sycl::aspect::fp64 not supported by the SYCL device."
                << std::endl;
      return;
    }
  }

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

// Hardcoded limits to avoid a "TernaryOpTestLauncher"
static constexpr int MIN_CLAMP = 5;
static constexpr int MAX_CLAMP = 10;

template <typename ValueT> void clamp_kernel(ValueT *a, ValueT *r) {
  *r = syclcompat::clamp(*a, static_cast<ValueT>(MIN_CLAMP),
                         static_cast<ValueT>(MAX_CLAMP));
}

template <typename ValueT> void test_clamp() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  ValueT op1 = static_cast<ValueT>(7);
  ValueT expect1 = static_cast<ValueT>(7);

  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<clamp_kernel<ValueT>>(op1, expect1);

  ValueT op2 = static_cast<ValueT>(MAX_CLAMP + 1);
  ValueT expect2 = static_cast<ValueT>(MAX_CLAMP);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<clamp_kernel<ValueT>>(op2, expect2);

  ValueT op3 = static_cast<ValueT>(MIN_CLAMP - 1);
  ValueT expect3 = static_cast<ValueT>(MIN_CLAMP);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<clamp_kernel<ValueT>>(op3, expect3);
}

int main() {
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_max);
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_min);

  // Basic testing of deduction to avoid combinatorial explosion
  test_syclcompat_max<double, float>();
  test_syclcompat_max<long, int>();
  test_syclcompat_min<double, float>();
  test_syclcompat_min<long, int>();

  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_fmin_nan);
  test_syclcompat_fmin_nan<double, float>();
  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_fmax_nan);
  test_syclcompat_fmax_nan<double, float>();

  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_pow);
  test_syclcompat_pow<float, int>();
  test_syclcompat_pow<double, int>();

  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_relu);
  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_cbrt);

  test_isnan();
  INSTANTIATE_ALL_TYPES(value_type_list, test_clamp);

  return 0;
}
