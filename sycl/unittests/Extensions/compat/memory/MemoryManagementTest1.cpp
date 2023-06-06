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
 *  MemoryManagementTest1.cpp
 *
 *  Description:
 *    memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ MemoryManagementTest1.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

#include "memory_check.hpp"

using namespace sycl::ext::oneapi::experimental;

TEST(Memory, memcpy) {

  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = (float *)compat::malloc(Num * sizeof(float));
  compat::memcpy((void *)d_A, (void *)h_A, N1 * sizeof(float));
  compat::memcpy((void *)(d_A + N1), (void *)h_B, (Num - N1) * sizeof(float));
  compat::memcpy((void *)h_C, (void *)d_A, Num * sizeof(float));
  compat::free((void *)d_A);

  compat::free(0);
  compat::free(NULL);
  compat::free(nullptr);

  // verify
  for (int i = 0; i < N1; i++) {
    EXPECT_EQ(h_A[i], h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    EXPECT_EQ(h_B[i], h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <typename T> class Memory_t : public ::testing::Test {};
using value_type_list =
    testing::Types<int, unsigned int, short, unsigned short, long,
                   unsigned long, long long, unsigned long long, float, double,
                   sycl::half>;
TYPED_TEST_SUITE(Memory_t, value_type_list);

template <typename T> class Fill_t : public ::testing::Test {
  void SetUp() {
    if (!compat::get_current_device().has(sycl::aspect::fp64) &&
        std::is_same_v<T, double>)
      GTEST_SKIP();
    if (!compat::get_current_device().has(sycl::aspect::fp16) &&
        std::is_same_v<T, sycl::half>)
      GTEST_SKIP();
  }
};
TYPED_TEST_SUITE(Fill_t, value_type_list);

TYPED_TEST(Memory_t, memcpy_t) {

  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  TypeParam *h_A = (TypeParam *)malloc(Num * sizeof(TypeParam));
  TypeParam *h_B = (TypeParam *)malloc(Num * sizeof(TypeParam));
  TypeParam *h_C = (TypeParam *)malloc(Num * sizeof(TypeParam));

  for (int i = 0; i < Num; i++) {
    h_A[i] = TypeParam(1);
    h_B[i] = TypeParam(2);
  }

  TypeParam *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = compat::malloc<TypeParam>(Num);
  compat::memcpy<TypeParam>(d_A, h_A, N1);
  compat::memcpy<TypeParam>((d_A + N1), h_B, (Num - N1));
  compat::memcpy<TypeParam>(h_C, d_A, Num);
  compat::free((void *)d_A);

  compat::free(0);
  compat::free(NULL);
  compat::free(nullptr);

  // verify
  for (int i = 0; i < N1; i++) {
    EXPECT_EQ(h_A[i], h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    EXPECT_EQ(h_B[i], h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

TEST(Memory, memset) {

  constexpr int Num = 10;
  int *h_A = (int *)malloc(Num * sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A = nullptr;

  d_A = (int *)compat::malloc(Num * sizeof(int));
  // hostA -> deviceA
  compat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(int));

  // set d_A[0,..., 6] = 0
  compat::memset((void *)d_A, 0, (Num - 3) * sizeof(int));

  // deviceA -> hostA
  compat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(int));

  compat::free((void *)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    EXPECT_EQ(h_A[i], 0);
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    EXPECT_EQ(h_A[i], 4);
  }

  free(h_A);
}

TYPED_TEST(Fill_t, fill) {

  constexpr int Num = 10;
  TypeParam *h_A = (TypeParam *)malloc(Num * sizeof(TypeParam));

  for (int i = 0; i < Num; i++) {
    h_A[i] = TypeParam(4);
  }

  TypeParam *d_A = nullptr;

  d_A = compat::malloc<TypeParam>(Num);
  // hostA -> deviceA
  compat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(TypeParam));

  // set d_A[0,..., 6] = 0
  compat::fill((void *)d_A, TypeParam(0), (Num - 3));

  // deviceA -> hostA
  compat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(TypeParam));

  compat::free((void *)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    EXPECT_EQ(h_A[i], TypeParam(0));
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    EXPECT_EQ(h_A[i], TypeParam(4));
  }

  free(h_A);
}

TEST(Memory, memcpy_q) {

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = (float *)compat::malloc(Num * sizeof(float), q);
  compat::memcpy((void *)d_A, (void *)h_A, N1 * sizeof(float), q);
  compat::memcpy((void *)(d_A + N1), (void *)h_B, (Num - N1) * sizeof(float),
                 q);
  compat::memcpy((void *)h_C, (void *)d_A, Num * sizeof(float), q);
  compat::free((void *)d_A, q);

  compat::free(0, q);
  compat::free(NULL, q);
  compat::free(nullptr, q);

  // verify
  for (int i = 0; i < N1; i++) {
    EXPECT_EQ(h_A[i], h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    EXPECT_EQ(h_B[i], h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

TYPED_TEST(Memory_t, memcpy_t_q) {

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  TypeParam *h_A = (TypeParam *)malloc(Num * sizeof(TypeParam));
  TypeParam *h_B = (TypeParam *)malloc(Num * sizeof(TypeParam));
  TypeParam *h_C = (TypeParam *)malloc(Num * sizeof(TypeParam));

  for (int i = 0; i < Num; i++) {
    h_A[i] = TypeParam(1);
    h_B[i] = TypeParam(2);
  }

  TypeParam *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = compat::malloc<TypeParam>(Num, q);
  compat::memcpy<TypeParam>(d_A, h_A, N1, q);
  compat::memcpy<TypeParam>((d_A + N1), h_B, (Num - N1), q);
  compat::memcpy<TypeParam>(h_C, d_A, Num, q);
  compat::free((void *)d_A, q);

  compat::free(0, q);
  compat::free(NULL, q);
  compat::free(nullptr, q);

  // verify
  for (int i = 0; i < N1; i++) {
    EXPECT_EQ(h_A[i], h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    EXPECT_EQ(h_B[i], h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

TEST(Memory, memset_q) {

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 10;
  int *h_A = (int *)malloc(Num * sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A = nullptr;

  d_A = (int *)compat::malloc(Num * sizeof(int), q);
  // hostA -> deviceA
  compat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(int), q);

  // set d_A[0,..., 6] = 0
  compat::memset((void *)d_A, 0, (Num - 3) * sizeof(int), q);

  // deviceA -> hostA
  compat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(int), q);

  compat::free((void *)d_A, q);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    EXPECT_EQ(h_A[i], 0);
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    EXPECT_EQ(h_A[i], 4);
  }

  free(h_A);
}

TYPED_TEST(Fill_t, fill_q) {

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 10;
  TypeParam *h_A = (TypeParam *)malloc(Num * sizeof(TypeParam));

  for (int i = 0; i < Num; i++) {
    h_A[i] = TypeParam(4);
  }

  TypeParam *d_A = nullptr;

  d_A = compat::malloc<TypeParam>(Num, q);
  // hostA -> deviceA
  compat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(TypeParam), q);

  // set d_A[0,..., 6] = 0
  compat::fill((void *)d_A, TypeParam(0), (Num - 3), q);

  // deviceA -> hostA
  compat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(TypeParam), q);

  compat::free((void *)d_A, q);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    EXPECT_EQ(h_A[i], TypeParam(0));
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    EXPECT_EQ(h_A[i], TypeParam(4));
  }

  free(h_A);
}
