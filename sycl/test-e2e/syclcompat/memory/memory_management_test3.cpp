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
 *  memory_management_test3.cpp
 *
 *  Description:
 *    memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_management_test3.cpp---------- -*- C++ -* ----===////
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

void test_free_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  float *d_A = (float *)syclcompat::malloc(sizeof(float));

  syclcompat::free(d_A);

  syclcompat::free(0);
  syclcompat::free(NULL);
  syclcompat::free(nullptr);
}

void test_free_memory_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  float *d_A = (float *)syclcompat::malloc(sizeof(float), q);
  syclcompat::free((void *)d_A, q);

  syclcompat::free(0, q);
  syclcompat::free(NULL, q);
  syclcompat::free(nullptr, q);
}

void test_memcpy_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int N1 = 1000;
  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  float *d_A = (float *)syclcompat::malloc(Num * sizeof(float));

  syclcompat::memcpy_async((void *)d_A, (void *)h_A, N1 * sizeof(float));
  syclcompat::memcpy_async((void *)(d_A + N1), (void *)h_B,
                           (Num - N1) * sizeof(float));
  syclcompat::memcpy_async((void *)h_C, (void *)d_A, Num * sizeof(float));

  syclcompat::wait();

  syclcompat::free((void *)d_A);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(fabs(h_A[i] - h_C[i]) <= 1e-5);
  }

  for (int i = N1; i < Num; i++) {
    assert(fabs(h_B[i] - h_C[i]) <= 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_memcpy_async_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  int Num = 5000;
  int N1 = 1000;
  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  float *d_A = (float *)syclcompat::malloc(Num * sizeof(float), q);
  syclcompat::memcpy_async((void *)d_A, (void *)h_A, N1 * sizeof(float), q);
  syclcompat::memcpy_async((void *)(d_A + N1), (void *)h_B,
                           (Num - N1) * sizeof(float), q);
  syclcompat::memcpy_async((void *)h_C, (void *)d_A, Num * sizeof(float), q);
  q.wait_and_throw();
  syclcompat::free((void *)d_A, q);

  // verify
  for (int i = 0; i < N1; i++) {
    assert(fabs(h_A[i] - h_C[i]) <= 1e-5);
  }

  for (int i = N1; i < Num; i++) {
    assert(fabs(h_B[i] - h_C[i]) <= 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_memcpy_async_pitched() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  size_t width = 6;
  size_t height = 8;
  float *h_data = nullptr;
  float *h_ref = nullptr;
  size_t h_pitch = sizeof(float) * width;
  h_data = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_data[i] = (float)i;

  h_ref = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_ref[i] = (float)i;

  // alloc device memory.
  size_t d_pitch;
  float *d_data =
      (float *)syclcompat::malloc(d_pitch, sizeof(float) * width, height);

  // copy to Device.
  syclcompat::memcpy_async(d_data, d_pitch, h_data, h_pitch,
                           sizeof(float) * width, height);

  // copy back to host.
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch,
                           sizeof(float) * width, height);

  syclcompat::get_default_queue().wait_and_throw();
  check(h_data, h_ref, width * height);

  // memset device data.
  syclcompat::memset_async(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch,
                           sizeof(float) * width, height);
  syclcompat::get_default_queue().wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width * height);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data);
}

void test_memcpy_async_pitched_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  size_t width = 6;
  size_t height = 8;
  float *h_data = nullptr;
  float *h_ref = nullptr;
  size_t h_pitch = sizeof(float) * width;
  h_data = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_data[i] = (float)i;

  h_ref = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_ref[i] = (float)i;

  // alloc device memory.
  size_t d_pitch;
  float *d_data =
      (float *)syclcompat::malloc(d_pitch, sizeof(float) * width, height, q);

  // copy to Device.
  syclcompat::memcpy_async(d_data, d_pitch, h_data, h_pitch,
                           sizeof(float) * width, height, q);

  // copy back to host.
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch,
                           sizeof(float) * width, height, q);
  q.wait_and_throw();
  check(h_data, h_ref, width * height);

  // memset device data.
  syclcompat::memset_async(d_data, d_pitch, 0x1, sizeof(float) * width, height,
                           q);

  // copy back to host
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch,
                           sizeof(float) * width, height, q);
  q.wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width * height);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data, q);
}

void test_memset_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 10;
  int *h_A = (int *)malloc(Num * sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A = (int *)syclcompat::malloc(Num * sizeof(int));
  // hostA -> deviceA
  syclcompat::memcpy_async((void *)d_A, (void *)h_A, Num * sizeof(int));

  // set d_A[0,..., 6] = 0
  syclcompat::memset_async((void *)d_A, 0, (Num - 3) * sizeof(int));

  // deviceA -> hostA
  syclcompat::memcpy_async((void *)h_A, (void *)d_A, Num * sizeof(int));

  syclcompat::get_default_queue().wait_and_throw();

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

void test_memset_async_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  int Num = 10;
  int *h_A = (int *)malloc(Num * sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A = (int *)syclcompat::malloc(Num * sizeof(int), q);
  // hostA -> deviceA
  syclcompat::memcpy_async((void *)d_A, (void *)h_A, Num * sizeof(int), q);

  // set d_A[0,..., 6] = 0
  syclcompat::memset_async((void *)d_A, 0, (Num - 3) * sizeof(int), q);

  // deviceA -> hostA
  syclcompat::memcpy_async((void *)h_A, (void *)d_A, Num * sizeof(int), q);
  q.wait_and_throw();
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

template <typename T> void test_memcpy_async_t_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  int Num = 5000;
  int N1 = 1000;
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
  syclcompat::memcpy_async<T>(d_A, h_A, N1, q);
  syclcompat::memcpy_async<T>((d_A + N1), h_B, (Num - N1), q);
  syclcompat::memcpy_async<T>(h_C, d_A, Num, q);
  q.wait_and_throw();
  syclcompat::free((void *)d_A, q);

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

template <typename T> void test_memcpy_async_t() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int N1 = 1000;
  T *h_A = (T *)malloc(Num * sizeof(T));
  T *h_B = (T *)malloc(Num * sizeof(T));
  T *h_C = (T *)malloc(Num * sizeof(T));

  for (int i = 0; i < Num; i++) {
    h_A[i] = static_cast<T>(1);
    h_B[i] = static_cast<T>(2);
  }

  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  T *d_A = syclcompat::malloc<T>(Num);
  syclcompat::memcpy_async<T>(d_A, h_A, N1);
  syclcompat::memcpy_async<T>((d_A + N1), h_B, (Num - N1));
  syclcompat::memcpy_async<T>(h_C, d_A, Num);

  syclcompat::wait();

  syclcompat::free((void *)d_A);

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

template <typename T> void test_fill_async() {
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
  syclcompat::fill_async((void *)d_A, static_cast<T>(0), (Num - 3));

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(T));

  syclcompat::get_default_queue().wait_and_throw();

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

template <typename T> void test_fill_async_q() {
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
  syclcompat::fill_async((void *)d_A, static_cast<T>(0), (Num - 3), q);

  // deviceA -> hostA
  syclcompat::memcpy((void *)h_A, (void *)d_A, Num * sizeof(T), q);

  q.wait_and_throw();

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

void test_constant_memcpy_async() {
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

  syclcompat::memcpy_async(d_A.get_ptr(), h_A, offset * sizeof(float));
  syclcompat::memcpy_async((char *)d_A.get_ptr() + offset * sizeof(float), h_B,
                           (size - offset) * sizeof(float));
  syclcompat::memcpy_async(h_C, d_A.get_ptr(), size * sizeof(float));
  syclcompat::memcpy_async(d_B.get_ptr(), d_A.get_ptr(),
                           offset * sizeof(float));
  syclcompat::memcpy_async((char *)d_A.get_ptr() + offset * sizeof(float), h_B,
                           (size - offset) * sizeof(float));
  syclcompat::memcpy_async((void *)h_C, (void *)d_A.get_ptr(),
                           size * sizeof(float));
  syclcompat::memcpy_async((void *)d_B.get_ptr(), (void *)d_A.get_ptr(),
                           offset * sizeof(float));
  syclcompat::memcpy_async(
      (char *)d_B.get_ptr() + offset * sizeof(float),
      (void *)((size_t)d_A.get_ptr() + offset * sizeof(float)),
      (size - offset) * sizeof(float));
  syclcompat::memcpy_async(h_D, d_B.get_ptr(), size * sizeof(float));
  syclcompat::get_default_queue().wait_and_throw();

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

void test_constant_memcpy_async_q() {
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

  syclcompat::memcpy_async(d_A.get_ptr(), h_A, offset * sizeof(float), q);

  syclcompat::memcpy_async((char *)d_A.get_ptr() + offset * sizeof(float), h_B,
                           (size - offset) * sizeof(float), q);
  syclcompat::memcpy_async(h_C, d_A.get_ptr(), size * sizeof(float), q);

  syclcompat::memcpy_async(d_B.get_ptr(), d_A.get_ptr(), offset * sizeof(float),
                           q);

  syclcompat::memcpy_async(
      (char *)d_B.get_ptr() + offset * sizeof(float),
      (void *)((size_t)d_A.get_ptr() + offset * sizeof(float)),
      (size - offset) * sizeof(float), q);

  syclcompat::memcpy_async(h_D, d_B.get_ptr(), size * sizeof(float), q);
  q.wait_and_throw();

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
  test_free_memory();
  test_free_memory_q();
  test_memcpy_async();
  test_memcpy_async_q();
  test_memcpy_async_pitched();
  test_memcpy_async_pitched_q();
  test_memset_async();
  test_memset_async_q();
  test_constant_memcpy_async();
  test_constant_memcpy_async_q();

  INSTANTIATE_ALL_TYPES(value_type_list, test_memcpy_async_t);
  INSTANTIATE_ALL_TYPES(value_type_list, test_memcpy_async_t_q);
  INSTANTIATE_ALL_TYPES(value_type_list, test_fill_async);
  INSTANTIATE_ALL_TYPES(value_type_list, test_fill_async_q);

  return 0;
}
