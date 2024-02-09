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
 *  atomic_comp_exchange.cpp
 *
 *  Description:
 *    atomic operations API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ Atomic.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: hip
// TODO: Remove unsupported after fixing
// https://github.com/intel/llvm/issues/12683
// UNSUPPORTED: accelerator

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <type_traits>

#include <sycl/sycl.hpp>

#include <syclcompat/atomic.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/id_query.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "atomic_fixt.hpp"

// Simple atomic kernels for testing
// In every case we test two API overloads, one taking an explicit runtime
// memory_order argument. We use `relaxed` in every case because these tests
// are *not* checking the memory_order semantics, just the API.
template <typename T>
inline void atomic_fetch_compare_inc_kernel(T *data, T operand) {
  syclcompat::atomic_fetch_compare_inc(data, operand);
}
template <typename T>
inline void atomic_fetch_compare_dec_kernel(T *data, T operand) {
  syclcompat::atomic_fetch_compare_dec(data, operand);
}
template <typename T1, typename T2>
inline void atomic_exchange_kernel(T1 *data, T2 operand) {
  syclcompat::atomic_exchange(data, operand);
}
template <typename T1, typename T2, typename T3>
inline void atomic_compare_exchange_strong_kernel(T1 *data, T2 expected,
                                                  T3 desired) {
  syclcompat::atomic_compare_exchange_strong(data, expected, desired);
}

void test_atomic_comp_inc() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{6};

  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(0, 6, 6);
  AtomicLauncher<atomic_fetch_compare_inc_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(1, 0, 6);
}

void test_atomic_comp_dec() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{6};

  AtomicLauncher<atomic_fetch_compare_dec_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(6, 0, 0);
  AtomicLauncher<atomic_fetch_compare_dec_kernel<unsigned int>, unsigned int>(
      grid, threads)
      .launch_test(0, 6, 11);
}

template <typename T> void test_atomic_exch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_exchange_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(1), static_cast<T>(1));
  AtomicLauncher<atomic_exchange_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
}

template <typename T> void test_atomic_ptr_exch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  using ValType = std::remove_pointer_t<T>;
  T ptr1 = (T)syclcompat::malloc(sizeof(ValType));
  T ptr2 = (T)syclcompat::malloc(sizeof(ValType));

  AtomicLauncher<atomic_exchange_kernel<T, T>, T>(grid, threads)
      .launch_test(ptr1, ptr2, ptr2);
  AtomicLauncher<atomic_exchange_kernel<T, T>, T>(grid, threads)
      .launch_test(ptr1, ptr1, ptr1);
  syclcompat::free(ptr1);
  syclcompat::free(ptr2);
}

template <typename T> void test_atomic_exch_strong() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_compare_exchange_strong_kernel<T, T, T>, T>(grid,
                                                                    threads)
      .launch_test(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0),
                   static_cast<T>(1));
  AtomicLauncher<atomic_compare_exchange_strong_kernel<T, T, T>, T>(grid,
                                                                    threads)
      .launch_test(static_cast<T>(0), static_cast<T>(0), static_cast<T>(1),
                   static_cast<T>(2));
}

template <typename T> void test_atomic_ptr_exch_strong() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  using ValType = std::remove_pointer_t<T>;
  T ptr1 = (T)syclcompat::malloc(sizeof(ValType));
  T ptr2 = (T)syclcompat::malloc(sizeof(ValType));
  T ptr3 = (T)syclcompat::malloc(sizeof(ValType));

  AtomicLauncher<atomic_compare_exchange_strong_kernel<T, T, T>, T>(grid,
                                                                    threads)
      .launch_test(ptr1, ptr2, ptr1, ptr2);
  AtomicLauncher<atomic_compare_exchange_strong_kernel<T, T, T>, T>(grid,
                                                                    threads)
      .launch_test(ptr1, ptr1, ptr2, ptr3);
  syclcompat::free(ptr1);
  syclcompat::free(ptr2);
  syclcompat::free(ptr3);
}

void test_atomic_exch_t1_t2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_exchange_kernel<float, int>, float>(grid, threads)
      .launch_test(static_cast<float>(0), static_cast<float>(1),
                   static_cast<int>(1));
  AtomicLauncher<atomic_exchange_kernel<float, int>, float>(grid, threads)
      .launch_test(static_cast<float>(0), static_cast<float>(0),
                   static_cast<int>(0));
}

void test_atomic_exch_strong_t1_t2_t3() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_compare_exchange_strong_kernel<float, int, unsigned>,
                 float>(grid, threads)
      .launch_test(static_cast<float>(0), static_cast<float>(1),
                   static_cast<int>(0), static_cast<unsigned>(1));
  AtomicLauncher<atomic_compare_exchange_strong_kernel<float, int, unsigned>,
                 float>(grid, threads)
      .launch_test(static_cast<float>(0), static_cast<float>(0),
                   static_cast<int>(1), static_cast<unsigned>(2));
}

int main() {
  INSTANTIATE_ALL_TYPES(atomic_value_type_list, test_atomic_exch);
  INSTANTIATE_ALL_TYPES(atomic_value_type_list, test_atomic_exch_strong);

  INSTANTIATE_ALL_TYPES(atomic_ptr_type_list, test_atomic_ptr_exch);
  INSTANTIATE_ALL_TYPES(atomic_ptr_type_list, test_atomic_ptr_exch_strong);

  test_atomic_comp_inc();
  test_atomic_comp_dec();
  test_atomic_exch_t1_t2();
  test_atomic_exch_strong_t1_t2_t3();
  return 0;
}
