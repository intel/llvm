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
 *  atomic_arith.cpp
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
template <typename T, bool orderArg = false>
inline void atomic_fetch_add_kernel(T *data, syclcompat::arith_t<T> operand) {
  if constexpr (orderArg) {
    syclcompat::atomic_fetch_add(data, operand, sycl::memory_order::relaxed);
  } else {
    syclcompat::atomic_fetch_add(data, operand);
  }
}
template <typename T, bool orderArg = false>
inline void atomic_fetch_sub_kernel(T *data, syclcompat::arith_t<T> operand) {
  if constexpr (orderArg) {
    syclcompat::atomic_fetch_sub(data, operand, sycl::memory_order::relaxed);
  } else {
    syclcompat::atomic_fetch_sub(data, operand);
  }
}

template <typename T> void test_atomic_arith() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};
  constexpr T sum = static_cast<T>(grid.x * threads.x);
  constexpr T init = static_cast<T>(0);
  constexpr T operand = static_cast<T>(1);

  AtomicLauncher<atomic_fetch_add_kernel<T>, T>(grid, threads)
      .launch_test(init, sum, operand);
  AtomicLauncher<atomic_fetch_sub_kernel<T>, T>(grid, threads)
      .launch_test(sum, init, operand);
  AtomicLauncher<atomic_fetch_add_kernel<T, true>, T>(grid, threads)
      .launch_test(init, sum, operand);
  AtomicLauncher<atomic_fetch_sub_kernel<T, true>, T>(grid, threads)
      .launch_test(sum, init, operand);
}

template <typename T> void test_atomic_ptr_arith() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  using ValType = std::remove_pointer_t<T>;

  T init = (T)syclcompat::malloc(sizeof(ValType));
  T final = init + (grid.x * threads.x);
  constexpr std::ptrdiff_t operand = static_cast<std::ptrdiff_t>(1);

  AtomicLauncher<atomic_fetch_add_kernel<T>, T>(grid, threads)
      .launch_test(init, final, operand);

  AtomicLauncher<atomic_fetch_sub_kernel<T>, T>(grid, threads)
      .launch_test(final, init, operand);

  AtomicLauncher<atomic_fetch_add_kernel<T, true>, T>(grid, threads)
      .launch_test(init, final, operand);

  AtomicLauncher<atomic_fetch_sub_kernel<T, true>, T>(grid, threads)
      .launch_test(final, init, operand);
  syclcompat::free(init);
}

int main() {
  INSTANTIATE_ALL_TYPES(atomic_value_type_list, test_atomic_arith);
  INSTANTIATE_ALL_TYPES(atomic_ptr_type_list, test_atomic_ptr_arith);
}
