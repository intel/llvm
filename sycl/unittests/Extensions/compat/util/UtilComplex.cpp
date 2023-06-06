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
 *  SYCL compatibility API
 *
 *  UtilComplex.cpp
 *
 *  Description:
 *    Complex operations tests
 **************************************************************************/

// The original source was under the license below:
//===-------------- UtilComplex.cpp --------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//
// test_feature:Util_cabs

#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

#include <cmath>

using namespace sycl::ext::oneapi::experimental;

template <typename T> bool check(T x, float e[], int &index) {
  float precison = 0.001f;
  if ((std::abs(x.x() - e[index++]) < precison) &&
      (std::abs(x.y() - e[index++]) < precison)) {
    return true;
  }
  return false;
}

template <> bool check<float>(float x, float e[], int &index) {
  float precison = 0.001f;
  if (std::abs(x - e[index++]) < precison) {
    return true;
  }
  return false;
}

template <> bool check<double>(double x, float e[], int &index) {
  float precison = 0.001f;
  if (std::abs(x - e[index++]) < precison) {
    return true;
  }
  return false;
}

// Class to launch a kernel and run a lambda on output data
template <auto F> class ComplexLauncher {
protected:
  int *result_;
  int cpu_result_{0};

public:
  ComplexLauncher() {
    result_ = (int *)compat::malloc_shared(sizeof(int));
    *result_ = 0;
  };
  ~ComplexLauncher() { compat::free(result_); }
  void launch() {
    if (!compat::get_current_device().has(sycl::aspect::fp64))
      GTEST_SKIP();
    F(&cpu_result_);                  // Run on host
    compat::launch<F>(1, 1, result_); // Run on device
    compat::wait();
    EXPECT_EQ(*result_, 1);
    EXPECT_EQ(cpu_result_, 1);
  }
};

void kernel_abs(int *result) {

  sycl::float2 f1, f2;
  sycl::double2 d1, d2;

  f1 = sycl::float2(1.8, -2.7);
  f2 = sycl::float2(-3.6, 4.5);
  d1 = sycl::double2(5.4, -6.3);
  d2 = sycl::double2(-7.2, 8.1);

  int index = 0;
  bool r = true;
  float expect[2] = {8.297590, 3.244996};

  auto a1 = compat::cabs(d1);
  r = r && check(a1, expect, index);

  auto a2 = compat::cabs(f1);
  r = r && check(a2, expect, index);

  *result = r;
}

void kernel_conj(int *result) {

  sycl::float2 f1, f2;
  sycl::double2 d1, d2;

  f1 = sycl::float2(1.8, -2.7);
  f2 = sycl::float2(-3.6, 4.5);
  d1 = sycl::double2(5.4, -6.3);
  d2 = sycl::double2(-7.2, 8.1);

  int index = 0;
  bool r = true;
  float expect[4] = {5.400000, 6.300000, 1.800000, 2.700000};

  auto a1 = compat::conj(d1);
  r = r && check(a1, expect, index);

  auto a2 = compat::conj(f1);
  r = r && check(a2, expect, index);

  *result = r;
}

void kernel_div(int *result) {

  sycl::float2 f1, f2;
  sycl::double2 d1, d2;

  f1 = sycl::float2(1.8, -2.7);
  f2 = sycl::float2(-3.6, 4.5);
  d1 = sycl::double2(5.4, -6.3);
  d2 = sycl::double2(-7.2, 8.1);

  int index = 0;
  bool r = true;
  float expect[4] = {-0.765517, 0.013793, -0.560976, 0.048780};

  auto a1 = compat::cdiv(d1, d2);
  r = r && check(a1, expect, index);

  auto a2 = compat::cdiv(f1, f2);
  r = r && check(a2, expect, index);

  *result = r;
}

void kernel_mul(int *result) {

  sycl::float2 f1, f2;
  sycl::double2 d1, d2;

  f1 = sycl::float2(1.8, -2.7);
  f2 = sycl::float2(-3.6, 4.5);
  d1 = sycl::double2(5.4, -6.3);
  d2 = sycl::double2(-7.2, 8.1);

  int index = 0;
  bool r = true;
  float expect[4] = {12.150000, 89.100000, 5.670001, 17.820000};

  auto a1 = compat::cmul(d1, d2);
  r = r && check(a1, expect, index);

  auto a2 = compat::cmul(f1, f2);
  r = r && check(a2, expect, index);

  *result = r;
}

TEST(Complex, abs) { ComplexLauncher<kernel_abs>().launch(); }
TEST(Complex, mul) { ComplexLauncher<kernel_mul>().launch(); }
TEST(Complex, div) { ComplexLauncher<kernel_div>().launch(); }
TEST(Complex, conj) { ComplexLauncher<kernel_conj>().launch(); }

TEST(Complex, DataType) {
  if (!std::is_same<compat::detail::DataType<float>::T2, float>::value)
    FAIL();
  if (!std::is_same<compat::detail::DataType<sycl::float2>::T2,
                    complex<float>>::value)
    FAIL();
}
