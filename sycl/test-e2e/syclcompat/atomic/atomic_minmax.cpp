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
 *  atomic_minmax.cpp
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
template <typename T1, typename T2>
inline void atomic_fetch_min_kernel(T1 *data, T2 operand, T2 operand0) {
  syclcompat::atomic_fetch_min(
      data, (syclcompat::global_id::x() == 0 ? operand0 : operand));
}
template <typename T1, typename T2>
inline void atomic_fetch_max_kernel(T1 *data, T2 operand, T2 operand0) {
  syclcompat::atomic_fetch_max(
      data, (syclcompat::global_id::x() == 0 ? operand0 : operand));
}

template <typename T> void test_atomic_minmax() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_fetch_min_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(100), static_cast<T>(1), static_cast<T>(200),
                   static_cast<T>(1));
  AtomicLauncher<atomic_fetch_max_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(100), static_cast<T>(200),
                   static_cast<T>(200), static_cast<T>(1));
}

template <typename T> void test_signed_atomic_minmax() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_fetch_min_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(-1), static_cast<T>(-4), static_cast<T>(-4),
                   static_cast<T>(100));
  AtomicLauncher<atomic_fetch_max_kernel<T, T>, T>(grid, threads)
      .launch_test(static_cast<T>(-40), static_cast<T>(-30),
                   static_cast<T>(-30), static_cast<T>(-100));
}

void test_signed_atomic_minmax_t1_t2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  AtomicLauncher<atomic_fetch_min_kernel<float, int>, float>(grid, threads)
      .launch_test(static_cast<float>(-1), static_cast<float>(-4),
                   static_cast<int>(-4), static_cast<int>(100));
  AtomicLauncher<atomic_fetch_max_kernel<float, int>, float>(grid, threads)
      .launch_test(static_cast<float>(-40), static_cast<float>(-30),
                   static_cast<int>(-30), static_cast<int>(-100));
}

int main() {
  INSTANTIATE_ALL_TYPES(atomic_value_type_list, test_atomic_minmax);
  INSTANTIATE_ALL_TYPES(signed_type_list, test_signed_atomic_minmax);
  test_signed_atomic_minmax_t1_t2();

  return 0;
}
