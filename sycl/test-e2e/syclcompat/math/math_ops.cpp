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

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/dims.hpp>
#include <syclcompat/math.hpp>

#include "../common.hpp"
#include "math_fixt.hpp"

template <typename ValueT, typename ValueU>
inline void max_kernel(ValueT *a, ValueU *b,
                       std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::max<ValueT, ValueU>(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_max() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  const ValueT op1 = static_cast<ValueT>(5);
  const ValueU op2 = static_cast<ValueU>(10);
  const std::common_type_t<ValueT, ValueU> res = static_cast<ValueU>(10);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<max_kernel<ValueT, ValueU>>(op1, op2, res);
}

template <typename ValueT, typename ValueU>
inline void min_kernel(ValueT *a, ValueU *b,
                       std::common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::min<ValueT,ValueU>(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_min() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  const ValueT op1 = static_cast<ValueT>(5);
  const ValueU op2 = static_cast<ValueU>(10);
  const std::common_type_t<ValueT, ValueU> res =
      static_cast<std::common_type_t<ValueT, ValueU>>(5);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<min_kernel<ValueT, ValueU>>(op1, op2, res);
}

template <typename ValueT, typename ValueU>
inline void fmin_nan_kernel(ValueT *a, ValueU *b,
                            container_common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::fmin_nan(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_fmin_nan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using ValueTU = std::common_type_t<ValueT, ValueU>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  const ValueT op1 = static_cast<ValueT>(5);
  const ValueU op2 = static_cast<ValueU>(10);
  ValueU op3 = sycl::nan(static_cast<unsigned int>(0));

  const ValueTU res =
      static_cast<ValueTU>(5);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ValueT, ValueU>>(op1, op2, res);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ValueT, ValueU>>(op1, op3, op3);
}

template <template <typename T, int Dim> typename ContainerT, typename ValueT, typename ValueU = ValueT>
void test_container_syclcompat_fmin_nan(){
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  using ValueTU = std::common_type_t<ValueT, ValueU>;
  using ContT = ContainerT<ValueT, 2>;
  using ContU = ContainerT<ValueU, 2>;
  using ContTU = ContainerT<ValueTU, 2>;

  const ContT op4 = {static_cast<ValueT>(5), static_cast<ValueT>(10)};
  const ContU op5 = {static_cast<ValueU>(10), static_cast<ValueU>(5)};
  const ContU op6 = {sycl::nan(static_cast<unsigned int>(0)), sycl::nan(static_cast<unsigned int>(0))};
  const ContTU op6_res = {sycl::nan(static_cast<unsigned int>(0)), sycl::nan(static_cast<unsigned int>(0))};

  const ContTU res2{static_cast<ValueTU>(5), static_cast<ValueTU>(5)};

  BinaryOpTestLauncher<ContT, ContU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ContT, ContU>>(op4, op5, res2);

  BinaryOpTestLauncher<ContT, ContU>(grid, threads)
      .template launch_test<fmin_nan_kernel<ContT, ContU>>(op4, op6, op6_res);
}

template <typename ValueT, typename ValueU>
inline void fmax_nan_kernel(ValueT *a, ValueU *b,
                            container_common_type_t<ValueT, ValueU> *r) {
  *r = syclcompat::fmax_nan(*a, *b);
}

template <typename ValueT, typename ValueU = ValueT>
void test_syclcompat_fmax_nan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using ValueTU = std::common_type_t<ValueT, ValueU>;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  const ValueT op1 = static_cast<ValueT>(5);
  const ValueU op2 = static_cast<ValueU>(10);
  ValueU op3 = sycl::nan(static_cast<unsigned int>(0));

  const ValueTU res =
      static_cast<ValueTU>(10);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ValueT, ValueU>>(op1, op2, res);

  BinaryOpTestLauncher<ValueT, ValueU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ValueT, ValueU>>(op1, op3, op3);
}

template <template <typename T, int Dim> typename ContainerT, typename ValueT, typename ValueU = ValueT>
void test_container_syclcompat_fmax_nan(){
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};

  using ValueTU = std::common_type_t<ValueT, ValueU>;
  using ContT = ContainerT<ValueT, 2>;
  using ContU = ContainerT<ValueU, 2>;
  using ContTU = ContainerT<ValueTU, 2>;

  const ContT op4 = {static_cast<ValueT>(5), static_cast<ValueT>(10)};
  const ContU op5 = {static_cast<ValueU>(10), static_cast<ValueU>(5)};
  const ContU op6 = {sycl::nan(static_cast<unsigned int>(0)), sycl::nan(static_cast<unsigned int>(0))};
  const ContTU op6_res = {sycl::nan(static_cast<unsigned int>(0)), sycl::nan(static_cast<unsigned int>(0))};

  const ContTU res2{static_cast<ValueTU>(10), static_cast<ValueTU>(10)};

  BinaryOpTestLauncher<ContT, ContU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ContT, ContU>>(op4, op5, res2);

  BinaryOpTestLauncher<ContT, ContU>(grid, threads)
      .template launch_test<fmax_nan_kernel<ContT, ContU>>(op4, op6, op6_res);
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
  const ValueT op1 = static_cast<ValueT>(3);
  const ValueU op2 = static_cast<ValueU>(3);
  const ValueT res = static_cast<ValueT>(27);

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
  const ValueT op1 = static_cast<ValueT>(3);
  const ValueT res1 = static_cast<ValueT>(3);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<relu_kernel<ValueT>>(op1, res1);

  const ValueT op2 = std::is_signed_v<ValueT> ? static_cast<ValueT>(-3)
                                              : static_cast<ValueT>(2);
  const ValueT res2 = std::is_signed_v<ValueT> ? static_cast<ValueT>(0)
                                               : static_cast<ValueT>(2);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<relu_kernel<ValueT>>(op2, res2);

  using ValueU = sycl::vec<ValueT, 2>;
  const ValueU op3{op1, op2};
  const ValueU res3{res1, res2};
  UnaryOpTestLauncher<ValueU>(grid, threads)
      .template launch_test<relu_kernel<ValueU>>(op3, res3);

  using ValueV = sycl::marray<ValueT, 2>;
  const ValueV op4{op1, op2};
  const ValueV res4{res1, res2};
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

  const ValueT op1 = static_cast<ValueT>(1);
  const ValueT res1 = static_cast<ValueT>(1);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<cbrt_kernel<ValueT>>(op1, res1);

  const ValueT op2 = static_cast<ValueT>(64);
  const ValueT res2 = static_cast<ValueT>(4);
  UnaryOpTestLauncher<ValueT>(grid, threads)
      .template launch_test<cbrt_kernel<ValueT>>(op2, res2);
}

template <typename T>
void isnan_kernel(T *a, T *r) {
  *r = syclcompat::isnan(*a);
}

template <template <typename, int> typename ContainerT, typename ValueT>
void test_isnan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using ContT = ContainerT<ValueT, 2>;
  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  ContT op1 = {sycl::nan(static_cast<unsigned int>(0)), 1.0f};
  // bool2 does not exist,1.0 and 0.0 floats are used for true
  // and false instead.
  ContT expect = {1.0, 0.0};

  UnaryOpTestLauncher<ContT>(grid, threads)
      .template launch_test<isnan_kernel<ContT>>(op1, expect);
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

template <template <typename T, int Dim> typename ContainerT, typename ValueT> void test_container_clamp() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{1};
  ValueT op1 = static_cast<ValueT>(7);
  ValueT expect1 = static_cast<ValueT>(7);

  ValueT op2 = static_cast<ValueT>(MAX_CLAMP + 1);
  ValueT expect2 = static_cast<ValueT>(MAX_CLAMP);

  using ContT = ContainerT<ValueT, 2>;
  const ContT op4{op1, op2};
  const ContT expect4{expect1, expect2};
  UnaryOpTestLauncher<ContT>(grid, threads)
      .template launch_test<clamp_kernel<ContT>>(op4, expect4);
}

int main() {
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_max);
  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_min);

  // Basic testing of deduction to avoid combinatorial explosion
  test_syclcompat_max<double, float>();
  test_syclcompat_max<long, int>();
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
  test_syclcompat_max<sycl::ext::oneapi::bfloat16, float>();
#endif

  test_syclcompat_min<double, float>();
  test_syclcompat_min<long, int>();
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
  test_syclcompat_min<sycl::ext::oneapi::bfloat16, float>();
#endif

  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_fmin_nan);
  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::vec, test_container_syclcompat_fmin_nan);
  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::marray, test_container_syclcompat_fmin_nan);
  test_syclcompat_fmin_nan<double, float>();
  test_container_syclcompat_fmin_nan<sycl::vec, float, double>();
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
  test_container_syclcompat_fmin_nan<sycl::vec, sycl::ext::oneapi::bfloat16, double>();
#endif

  INSTANTIATE_ALL_TYPES(fp_type_list, test_syclcompat_fmax_nan);
  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::vec, test_container_syclcompat_fmax_nan);
  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::marray, test_container_syclcompat_fmax_nan);
  test_syclcompat_fmax_nan<double, float>();
  test_container_syclcompat_fmax_nan<sycl::vec, float, double>();
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
  test_container_syclcompat_fmax_nan<sycl::vec, sycl::ext::oneapi::bfloat16, double>();
#endif

  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_pow);
  test_syclcompat_pow<float, int>();
  test_syclcompat_pow<double, int>();

  INSTANTIATE_ALL_TYPES(value_type_list, test_syclcompat_relu);
  INSTANTIATE_ALL_TYPES(fp_type_list_no_bfloat16, test_syclcompat_cbrt);

  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::vec, test_isnan);
  INSTANTIATE_ALL_CONTAINER_TYPES(fp_type_list, sycl::marray, test_isnan);

  INSTANTIATE_ALL_TYPES(value_type_list, test_clamp);
  INSTANTIATE_ALL_CONTAINER_TYPES(vec_type_list, sycl::vec, test_container_clamp);
  INSTANTIATE_ALL_CONTAINER_TYPES(marray_type_list, sycl::marray, test_container_clamp);

  return 0;
}
