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
 *  math.cpp
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

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/half_type.hpp>
#include <syclcompat/math.hpp>

// These tests only check the API, not the functionality itself.


void test_compare_half() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half h;
  syclcompat::compare(h, h, std::equal_to<>());
}

void test_compare_half2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half2 h2;
  syclcompat::compare(h2, h2, std::equal_to<>());
}
void test_unordered_compare_half() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half h;
  syclcompat::unordered_compare(h, h, std::equal_to<>());
}
void test_unordered_compare_half2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half2 h2;
  syclcompat::unordered_compare(h2, h2, std::equal_to<>());
}
void test_compare_both() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half2 h2;
  syclcompat::compare_both(h2, h2, std::equal_to<>());
}
void test_unordered_compare_both() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half2 h2;
  syclcompat::unordered_compare_both(h2, h2, std::equal_to<>());
}
void test_isnan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::half2 h2;
  syclcompat::isnan(h2);
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
  test_length();
  test_compare_half();
  test_compare_half2();
  test_unordered_compare_half();
  test_unordered_compare_half2();
  test_compare_both();
  test_unordered_compare_both();
  test_isnan();
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
