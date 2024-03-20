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
 *  util_matrix_mem_copy_test.cpp
 *
 *  Description:
 *    matrix_mem_copy tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilMatrixMemCopyTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

#define M 3
#define N 2

void test_matrix_mem_copy_1() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();
  float *devPtrA;
  devPtrA = (float *)sycl::malloc_device(M * N * sizeof(float), *q_ct1);
  float host_a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float host_b[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  float host_c[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  syclcompat::detail::matrix_mem_copy((void *)devPtrA, (void *)host_a, M, M, M,
                                      N, sizeof(float));
  syclcompat::detail::matrix_mem_copy((void *)host_b, (void *)devPtrA, M, M, M,
                                      N, sizeof(float));

  for (int i = 0; i < M * N; i++) {
    assert(fabs(host_b[i] - host_c[i]) <= 1e-5);
  }

  // Because to_ld == from_ld, matrix_mem_copy just do one copy.
  // All padding data is also copied except the last padding.
  float host_d[6] = {-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f};
  float host_e[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, -2.0f};
  syclcompat::detail::matrix_mem_copy(
      (void *)host_d, (void *)devPtrA, M /*to_ld*/, M /*from_ld*/,
      M - 1 /*rows*/, N /*cols*/, sizeof(float));

  for (int i = 0; i < M * N; i++) {
    assert(fabs(host_d[i] - host_e[i]) <= 1e-5);
  }

  sycl::free(devPtrA, *q_ct1);
}

void test_matrix_mem_copy_2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  float *devPtrA;
  devPtrA = (float *)sycl::malloc_device(M * N * sizeof(float), *q_ct1);
  float host_a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float host_b[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  float host_c[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  syclcompat::detail::matrix_mem_copy(devPtrA, host_a, M, M, M, N);
  syclcompat::detail::matrix_mem_copy(host_b, devPtrA, M, M, M, N);

  for (int i = 0; i < M * N; i++) {
    assert(fabs(host_b[i] - host_c[i]) <= 1e-5);
  }

  float host_d[4] = {-2.0f, -2.0f, -2.0f, -2.0f};
  float host_e[4] = {1.0f, 2.0f, 4.0f, 5.0f};
  syclcompat::detail::matrix_mem_copy(host_d, devPtrA, M - 1 /*to_ld*/,
                                      M /*from_ld*/, M - 1 /*rows*/,
                                      N /*cols*/);

  for (int i = 0; i < (M - 1) * (N - 1); i++) {
    assert(fabs(host_d[i] - host_e[i]) <= 1e-5);
  }

  sycl::free(devPtrA, *q_ct1);
}

int main() {
  test_matrix_mem_copy_1();
  test_matrix_mem_copy_2();

  return 0;
}
