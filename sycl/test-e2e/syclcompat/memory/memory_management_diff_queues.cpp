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
 *  memory_management_diff_queue.cpp
 *
 *  Description:
 *    memory operations tests for operations when changing the default queue
 **************************************************************************/

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "memory_common.hpp"
#include "memory_fixt.hpp"

void test_memcpy() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q{{sycl::property::queue::in_order()}};

  constexpr int ELEMENTS = 5000;
  constexpr int N1 = 1000;
  float *h_A = (float *)malloc(ELEMENTS * sizeof(float));
  float *h_B = (float *)malloc(ELEMENTS * sizeof(float));
  float *h_C = (float *)malloc(ELEMENTS * sizeof(float));

  for (int i = 0; i < ELEMENTS; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = (float *)syclcompat::malloc(ELEMENTS * sizeof(float), q);
  syclcompat::memcpy((void *)d_A, (void *)h_A, N1 * sizeof(float), q);

  syclcompat::set_default_queue(q);
  syclcompat::memcpy((void *)(d_A + N1), (void *)h_B,
                     (ELEMENTS - N1) * sizeof(float));

  syclcompat::memcpy((void *)h_C, (void *)d_A, ELEMENTS * sizeof(float));

  // verify
  for (int i = 0; i < N1; i++) {
    assert(h_A[i] == h_C[i]);
  }

  for (int i = N1; i < ELEMENTS; i++) {
    assert(h_B[i] == h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
  syclcompat::free((void *)d_A);
}

void test_memset() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q{{sycl::property::queue::in_order()}};

  constexpr int PORTION = 5;
  constexpr int ELEMENTS = PORTION * 3;

  int *h_A = (int *)malloc(ELEMENTS * sizeof(int));
  for (int i = 0; i < ELEMENTS; i++) {
    h_A[i] = 4;
  }

  int *d_A = nullptr;

  d_A = (int *)syclcompat::malloc(ELEMENTS * sizeof(int));
  // hostA -> deviceA
  syclcompat::memcpy((void *)d_A, (void *)h_A, ELEMENTS * sizeof(int), q);

  // set d_A[0,..., PORTION - 1] = 0
  syclcompat::memset((void *)d_A, 0, PORTION * sizeof(int), q);

  syclcompat::set_default_queue(q);
  // set d_A[PORTION,..., 2 * PORTION - 1] = 0x01010101
  syclcompat::memset((void *)(d_A + PORTION), 1, PORTION * sizeof(int));
  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, ELEMENTS * sizeof(int));

  // check d_A[0,..., PORTION - 1] = 0
  for (int i = 0; i < PORTION; i++) {
    assert(h_A[i] == 0);
  }

  // check d_A[PORTION,..., 2 * PORTION - 1] = 0x01010101
  for (int i = PORTION; i < (2 * PORTION - 1); i++) {
    assert(h_A[i] == 0x01010101);
  }

  // check d_A[2 * PORTION,..., ELEMENTS] = 4
  for (int i = 2 * PORTION; i < ELEMENTS; i++) {
    assert(h_A[i] == 4);
  }

  free(h_A);
  syclcompat::free((void *)d_A);
}

int main() {
  test_memcpy();
  test_memset();

  return 0;
}
