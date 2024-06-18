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
 *  atomic_logic.cpp
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

// FIXME: This isn't entirely true, it's not supported in hardware without
// seq_acq but the assertion is done at compile-time for AMDGPU, which causes CI
// to fail. The same applies to each test within this directory
// UNSUPPORTED: hip

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <type_traits>

#include <sycl/detail/core.hpp>

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
template <typename T1, typename T2>
void atomic_fetch_and_kernel(T1 *data, T2 operand, T2 operand0) {
  syclcompat::atomic_fetch_and(
      data, (syclcompat::global_id::x() == 0 ? operand0 : operand));
}
template <typename T1, typename T2>
void atomic_fetch_or_kernel(T1 *data, T2 operand, T2 operand0) {
  syclcompat::atomic_fetch_or(
      data, (syclcompat::global_id::x() == 0 ? operand0 : operand));
}
template <typename T1, typename T2>
void atomic_fetch_xor_kernel(T1 *data, T2 operand, T2 operand0) {
  syclcompat::atomic_fetch_xor(
      data, (syclcompat::global_id::x() == 0 ? operand0 : operand));
}

template <typename T> void test_atomic_and() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
                   static_cast<T>(0));

  // All 1 -> 1
  AtomicLauncher<atomic_fetch_and_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
                   static_cast<T>(1));
  // Most 1, one 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(0), static_cast<T>(1),
                   static_cast<T>(0));
}

template <typename T> void test_atomic_or() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_or_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
                   static_cast<T>(0));
  // All 1 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
                   static_cast<T>(1));
  // Most 1, one 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
                   static_cast<T>(0));
  // Init 1, all 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(1), static_cast<T>(0),
                   static_cast<T>(0));
}

template <typename T> void test_atomic_xor() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{2}; // 2 threads, 3 values inc. init

  // 000 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
                   static_cast<T>(0));
  // 111 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
                   static_cast<T>(1));
  // 110 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(1), static_cast<T>(0), static_cast<T>(1),
                   static_cast<T>(0));
  // 010 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(0), static_cast<T>(1), static_cast<T>(1),
                   static_cast<T>(0));
}

void test_atomic_and_t1_t2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  using data_t = long;
  using operand_t = unsigned int;

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(0), static_cast<data_t>(0),
                   static_cast<operand_t>(0), static_cast<operand_t>(0));

  // All 1 -> 1
  AtomicLauncher<atomic_fetch_and_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(1),
                   static_cast<operand_t>(1), static_cast<operand_t>(1));
  // Most 1, one 0 -> 0
  AtomicLauncher<atomic_fetch_and_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(0),
                   static_cast<operand_t>(1), static_cast<operand_t>(0));
}

void test_atomic_or_t1_t2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  using data_t = long;
  using operand_t = unsigned int;

  // All 0 -> 0
  AtomicLauncher<atomic_fetch_or_kernel<data_t, operand_t>, data_t>(grid,
                                                                    threads)
      .launch_test(static_cast<data_t>(0), static_cast<data_t>(0),
                   static_cast<operand_t>(0), static_cast<operand_t>(0));
  // All 1 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<data_t, operand_t>, data_t>(grid,
                                                                    threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(1),
                   static_cast<operand_t>(1), static_cast<operand_t>(1));
  // Most 1, one 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<data_t, operand_t>, data_t>(grid,
                                                                    threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(1),
                   static_cast<operand_t>(1), static_cast<operand_t>(0));
  // Init 1, all 0 -> 1
  AtomicLauncher<atomic_fetch_or_kernel<data_t, operand_t>, data_t>(grid,
                                                                    threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(1),
                   static_cast<operand_t>(0), static_cast<operand_t>(0));
}

void test_atomic_xor_t1_t2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{2}; // 2 threads, 3 values inc. init

  using data_t = long;
  using operand_t = unsigned int;

  // 000 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(0), static_cast<data_t>(0),
                   static_cast<operand_t>(0), static_cast<operand_t>(0));
  // 111 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(1),
                   static_cast<operand_t>(1), static_cast<operand_t>(1));
  // 110 -> 0
  AtomicLauncher<atomic_fetch_xor_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(1), static_cast<data_t>(0),
                   static_cast<operand_t>(1), static_cast<operand_t>(0));
  // 010 -> 1
  AtomicLauncher<atomic_fetch_xor_kernel<data_t, operand_t>, data_t>(grid,
                                                                     threads)
      .launch_test(static_cast<data_t>(0), static_cast<data_t>(1),
                   static_cast<operand_t>(1), static_cast<operand_t>(0));
}

int main() {
  INSTANTIATE_ALL_TYPES(integral_type_list, test_atomic_and);
  INSTANTIATE_ALL_TYPES(integral_type_list, test_atomic_or);
  INSTANTIATE_ALL_TYPES(integral_type_list, test_atomic_xor);

  // Avoid combinatorial explosion by only testing the interface
  test_atomic_and_t1_t2();
  test_atomic_or_t1_t2();
  test_atomic_xor_t1_t2();

  return 0;
}
