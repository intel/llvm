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
 *  memory_management_test1.cpp
 *
 *  Description:
 *    memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_management_test1.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "memory_common.hpp"
#include "memory_fixt.hpp"

void test_memcpy() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

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
  d_A = (float *)syclcompat::malloc(Num * sizeof(float));
  syclcompat::memcpy((void *)d_A, (void *)h_A, N1 * sizeof(float));
  syclcompat::memcpy((void *)(d_A + N1), (void *)h_B,
                     (Num - N1) * sizeof(float));
  syclcompat::memcpy((void *)h_C, (void *)d_A, Num * sizeof(float));
  syclcompat::free((void *)d_A);

  syclcompat::free(0);
  syclcompat::free(NULL);
  syclcompat::free(nullptr);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(h_A[i] == h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    assert(h_B[i] == h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_memcpy_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

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
  d_A = (float *)syclcompat::malloc(Num * sizeof(float), q);
  syclcompat::memcpy((void *)d_A, (void *)h_A, N1 * sizeof(float), q);
  syclcompat::memcpy((void *)(d_A + N1), (void *)h_B,
                     (Num - N1) * sizeof(float), q);
  syclcompat::memcpy((void *)h_C, (void *)d_A, Num * sizeof(float), q);
  syclcompat::free((void *)d_A, q);

  syclcompat::free(0, q);
  syclcompat::free(NULL, q);
  syclcompat::free(nullptr, q);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(h_A[i] == h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    assert(h_B[i] == h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <size_t memset_size_bits = 8> void test_memset_impl() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  // ValueT -> int for memset and memset_d32, short for memset_d16.
  using ValueT = std::conditional_t<
      memset_size_bits == 8 || memset_size_bits == 32, int,
      std::conditional_t<memset_size_bits == 16, short, void>>;
  static_assert(!std::is_void_v<ValueT>,
                "memset tests only work for 8, 16 and 32 bits");

  constexpr int Num = 10;
  ValueT *h_A = (ValueT *)malloc(Num * sizeof(ValueT));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  ValueT *d_A = (ValueT *)syclcompat::malloc(Num * sizeof(ValueT));
  // hostA -> deviceA
  syclcompat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(ValueT));

  // set d_A[0,..., 6] = 0
  if constexpr (memset_size_bits == 8)
    syclcompat::memset((void *)d_A, 0, (Num - 3) * sizeof(ValueT));
  else if constexpr (memset_size_bits == 16)
    syclcompat::memset_d16((void *)d_A, 0, (Num - 3));
  else if constexpr (memset_size_bits == 32)
    syclcompat::memset_d32((void *)d_A, 0, (Num - 3));

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(ValueT));

  syclcompat::free((void *)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    assert(h_A[i] == 0);
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    assert(h_A[i] == 4);
  }

  free(h_A);
}

template <size_t memset_size_bits = 8> void test_memset_q_impl() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  // ValueT -> int for memset and memset_d32, short for memset_d16.
  using ValueT = std::conditional_t<
      memset_size_bits == 8 || memset_size_bits == 32, int,
      std::conditional_t<memset_size_bits == 16, short, void>>;
  static_assert(!std::is_void_v<ValueT>,
                "memset tests only work for 8, 16 and 32 bits");

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 10;
  ValueT *h_A = (ValueT *)malloc(Num * sizeof(ValueT));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  ValueT *d_A = (ValueT *)syclcompat::malloc(Num * sizeof(ValueT), q);
  // hostA -> deviceA
  syclcompat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(ValueT), q);

  // set d_A[0,..., 6] = 0
  if constexpr (memset_size_bits == 8)
    syclcompat::memset((void *)d_A, 0, (Num - 3) * sizeof(ValueT), q);
  else if constexpr (memset_size_bits == 16)
    syclcompat::memset_d16((void *)d_A, 0, (Num - 3), q);
  else if constexpr (memset_size_bits == 32)
    syclcompat::memset_d32((void *)d_A, 0, (Num - 3), q);

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(ValueT), q);

  syclcompat::free((void *)d_A, q);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    assert(h_A[i] == 0);
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    assert(h_A[i] == 4);
  }

  free(h_A);
}

void test_memset() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 8;
  test_memset_impl<memset_size_in_bits>();
}

void test_memset_d16() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 16;
  test_memset_impl<memset_size_in_bits>();
}

void test_memset_d32() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 32;
  test_memset_impl<memset_size_in_bits>();
}

void test_memset_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 8;
  test_memset_q_impl<memset_size_in_bits>();
}

void test_memset_d16_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 16;
  test_memset_q_impl<memset_size_in_bits>();
}

void test_memset_d32_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  constexpr size_t memset_size_in_bits = 32;
  test_memset_q_impl<memset_size_in_bits>();
}

template <typename T> void test_memcpy_t() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  T *h_A = (T *)malloc(Num * sizeof(T));
  T *h_B = (T *)malloc(Num * sizeof(T));
  T *h_C = (T *)malloc(Num * sizeof(T));

  for (int i = 0; i < Num; i++) {
    h_A[i] = static_cast<T>(1);
    h_B[i] = static_cast<T>(2);
  }

  T *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = syclcompat::malloc<T>(Num);
  syclcompat::memcpy<T>(d_A, h_A, N1);
  syclcompat::memcpy<T>((d_A + N1), h_B, (Num - N1));
  syclcompat::memcpy<T>(h_C, d_A, Num);
  syclcompat::free((void *)d_A);

  syclcompat::free(0);
  syclcompat::free(NULL);
  syclcompat::free(nullptr);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(h_A[i] == h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    assert(h_B[i] == h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <typename T> void test_memcpy_t_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 5000;
  constexpr int N1 = 1000;
  T *h_A = (T *)malloc(Num * sizeof(T));
  T *h_B = (T *)malloc(Num * sizeof(T));
  T *h_C = (T *)malloc(Num * sizeof(T));

  for (int i = 0; i < Num; i++) {
    h_A[i] = static_cast<T>(1);
    h_B[i] = static_cast<T>(2);
  }

  T *d_A = nullptr;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = syclcompat::malloc<T>(Num, q);
  syclcompat::memcpy<T>(d_A, h_A, N1, q);
  syclcompat::memcpy<T>((d_A + N1), h_B, (Num - N1), q);
  syclcompat::memcpy<T>(h_C, d_A, Num, q);
  syclcompat::free((void *)d_A, q);

  syclcompat::free(0, q);
  syclcompat::free(NULL, q);
  syclcompat::free(nullptr, q);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(h_A[i] == h_C[i]);
  }

  for (int i = N1; i < Num; i++) {
    assert(h_B[i] == h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <typename T> void test_fill() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  bool skip = should_skip<T>(syclcompat::get_current_device());
  if (skip) // Unsupported aspect
    return;

  constexpr int Num = 10;
  T *h_A = (T *)malloc(Num * sizeof(T));

  for (int i = 0; i < Num; i++) {
    h_A[i] = static_cast<T>(4);
  }

  T *d_A = nullptr;

  d_A = syclcompat::malloc<T>(Num);
  // hostA -> deviceA
  syclcompat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(T));

  // set d_A[0,..., 6] = 0
  syclcompat::fill((void *)d_A, static_cast<T>(0), (Num - 3));

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(T));

  syclcompat::free((void *)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    assert(h_A[i] == static_cast<T>(0));
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    assert(h_A[i] == static_cast<T>(4));
  }

  free(h_A);
}

template <typename T> void test_fill_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  bool skip = should_skip<T>(syclcompat::get_current_device());
  if (skip) // Unsupported aspect
    return;
  sycl::queue q{{sycl::property::queue::in_order()}};
  constexpr int Num = 10;
  T *h_A = (T *)malloc(Num * sizeof(T));

  for (int i = 0; i < Num; i++) {
    h_A[i] = static_cast<T>(4);
  }

  T *d_A = nullptr;

  d_A = syclcompat::malloc<T>(Num, q);
  // hostA -> deviceA
  syclcompat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(T), q);

  // set d_A[0,..., 6] = 0
  syclcompat::fill((void *)d_A, static_cast<T>(0), (Num - 3), q);

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(T), q);

  syclcompat::free((void *)d_A, q);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    assert(h_A[i] == static_cast<T>(0));
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    assert(h_A[i] == static_cast<T>(4));
  }

  free(h_A);
}

void test_constant_memcpy() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr size_t size = 2000;
  constexpr size_t offset = 1000;

  syclcompat::constant_memory<float, 1> d_A(size);
  syclcompat::constant_memory<float, 1> d_B(size);

  float *h_A = (float *)malloc(size / 2 * sizeof(float));
  float *h_B = (float *)malloc(size / 2 * sizeof(float));
  float *h_C = (float *)malloc(size * sizeof(float));
  float *h_D = (float *)malloc(size * sizeof(float));

  for (int i = 0; i < size / 2; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..999] -> deviceA[1000..1999]
  // deviceA[0..1999] -> hostC[0..1999]
  // deviceA[0..999] -> deviceB[0..999]
  // deviceA[1000..1999] -> deviceB[1000..1999]
  // deviceB[0..1999] -> hostD[0..1999]

  syclcompat::memcpy(d_A.get_ptr(), h_A, offset * sizeof(float));
  syclcompat::memcpy((char *)d_A.get_ptr() + offset * sizeof(float), h_B,
                     (size - offset) * sizeof(float));
  syclcompat::memcpy(h_C, d_A.get_ptr(), size * sizeof(float));
  syclcompat::memcpy(d_B.get_ptr(), d_A.get_ptr(), offset * sizeof(float));
  syclcompat::memcpy((char *)d_B.get_ptr() + offset * sizeof(float),
                     (void *)((size_t)d_A.get_ptr() + offset * sizeof(float)),
                     (size - offset) * sizeof(float));
  syclcompat::memcpy(h_D, d_B.get_ptr(), size * sizeof(float));

  // verify hostD
  for (int i = 0; i < offset; i++) {
    assert(fabs(h_A[i] - h_D[i]) <= 1e-5);
  }

  for (int i = offset; i < size; i++) {
    assert(fabs(h_B[i - offset] - h_D[i]) <= 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_D);
}

void test_constant_memcpy_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};

  constexpr size_t size = 2000;
  constexpr size_t offset = 1000;
  syclcompat::constant_memory<float, 1> d_A(size, q);
  syclcompat::constant_memory<float, 1> d_B(size, q);

  float *h_A = (float *)malloc(size / 2 * sizeof(float));
  float *h_B = (float *)malloc(size / 2 * sizeof(float));
  float *h_C = (float *)malloc(size * sizeof(float));
  float *h_D = (float *)malloc(size * sizeof(float));

  for (int i = 0; i < size / 2; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..999] -> deviceA[1000..1999]
  // deviceA[0..1999] -> hostC[0..1999]
  // deviceA[0..999] -> deviceB[0..999]
  // deviceA[1000..1999] -> deviceB[1000..1999]
  // deviceB[0..1999] -> hostD[0..1999]

  syclcompat::memcpy(d_A.get_ptr(), h_A, offset * sizeof(float), q);

  syclcompat::memcpy((char *)d_A.get_ptr() + offset * sizeof(float), h_B,
                     (size - offset) * sizeof(float), q);
  syclcompat::memcpy(h_C, d_A.get_ptr(), size * sizeof(float), q);

  syclcompat::memcpy(d_B.get_ptr(), d_A.get_ptr(), offset * sizeof(float), q);

  syclcompat::memcpy((char *)d_B.get_ptr() + offset * sizeof(float),
                     (void *)((size_t)d_A.get_ptr() + offset * sizeof(float)),
                     (size - offset) * sizeof(float), q);

  syclcompat::memcpy(h_D, d_B.get_ptr(), size * sizeof(float), q);

  // verify hostD
  for (int i = 0; i < offset; i++) {
    assert(fabs(h_A[i] - h_D[i]) <= 1e-5);
  }

  for (int i = offset; i < size; i++) {
    assert(fabs(h_B[i - offset] - h_D[i]) <= 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_D);
}

int main() {
  test_memcpy();
  test_memcpy_q();
  test_memset();
  test_memset_q();
  test_memset_d16();
  test_memset_d16_q();
  test_memset_d32();
  test_memset_d32_q();
  test_constant_memcpy();
  test_constant_memcpy_q();

  INSTANTIATE_ALL_TYPES(value_type_list, test_memcpy_t);
  INSTANTIATE_ALL_TYPES(value_type_list, test_memcpy_t_q);
  INSTANTIATE_ALL_TYPES(value_type_list, test_fill);
  INSTANTIATE_ALL_TYPES(value_type_list, test_fill_q);

  return 0;
}
