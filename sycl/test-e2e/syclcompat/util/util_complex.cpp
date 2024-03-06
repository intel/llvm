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
 *  util_complex.cpp
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

// REQUIRES: aspect-fp64
// REQUIRES: usm_shared_allocations
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <complex>

#include <iostream>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

template <typename T> bool check(T x, float e[], int &index) {
  float precision = 0.001f;
  if ((x.x() - e[index] < precision) && (x.x() - e[index] > -precision) &&
      (x.y() - e[index + 1] < precision) &&
      (x.y() - e[index + 1] > -precision)) {
    index += 2;
    return true;
  }
  return false;
}

template <> bool check<float>(float x, float e[], int &index) {
  float precision = 0.001f;
  if ((x - e[index] < precision) && (x - e[index] > -precision)) {
    index++;
    return true;
  }
  return false;
}

template <> bool check<double>(double x, float e[], int &index) {
  float precision = 0.001f;
  if ((x - e[index] < precision) && (x - e[index] > -precision)) {
    index++;
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
    result_ = (int *)syclcompat::malloc_shared(sizeof(int));
    *result_ = 0;
  };
  ~ComplexLauncher() { syclcompat::free(result_); }
  void launch() {
    F(&cpu_result_);                      // Run on host
    syclcompat::launch<F>(1, 1, result_); // Run on device
    syclcompat::wait();
    assert(*result_ == 1);
    assert(cpu_result_ == 1);
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

  auto a1 = syclcompat::cabs(d1);
  r = r && check(a1, expect, index);

  auto a2 = syclcompat::cabs(f1);
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

  auto a1 = syclcompat::conj(d1);
  r = r && check(a1, expect, index);

  auto a2 = syclcompat::conj(f1);
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

  auto a1 = syclcompat::cdiv(d1, d2);
  r = r && check(a1, expect, index);

  auto a2 = syclcompat::cdiv(f1, f2);
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

  auto a1 = syclcompat::cmul(d1, d2);
  r = r && check(a1, expect, index);

  auto a2 = syclcompat::cmul(f1, f2);
  r = r && check(a2, expect, index);

  *result = r;
}

void test_abs() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  ComplexLauncher<kernel_abs>().launch();
}
void test_mul() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  ComplexLauncher<kernel_mul>().launch();
}
void test_div() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  ComplexLauncher<kernel_div>().launch();
}
void test_conj() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  ComplexLauncher<kernel_conj>().launch();
}

int main() {
  test_abs();
  test_mul();
  test_div();
  test_conj();

  return 0;
}
