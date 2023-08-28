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

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>
#include <syclcompat/atomic.hpp>

#include "../common.hpp"
#include "atomic_fixt.hpp"

template <typename T> void atomic_ref_value_kernel(T *atom_arr) {
  syclcompat::atomic<T> a{static_cast<T>(0)};
  T temp1 = static_cast<T>(3);
  T temp2 = static_cast<T>(4);

  // atomic store
  a.store(static_cast<T>(1));

  // atomic load
  atom_arr[0] = a.load();

  // atomic exchange
  atom_arr[1] = a.exchange(static_cast<T>(3));

  // atomic compare_exchange_weak
  atom_arr[2] = a.load();
  a.compare_exchange_weak(temp1, static_cast<T>(4));

  // atomic compare_exchange_strong
  atom_arr[3] = a.load();
  a.compare_exchange_strong(temp2, static_cast<T>(8));

  // atomic fetch_add
  atom_arr[4] = a.fetch_add(static_cast<T>(1));

  // atomic fetch_sub
  atom_arr[5] = a.fetch_sub(static_cast<T>(-1));
}

template <typename T> void atomic_class_value() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr size_t numBlocks = 64;
  constexpr size_t numThreads = 256;
  constexpr size_t numData = 6;

  AtomicClassLauncher<atomic_ref_value_kernel<T>, T>(numBlocks, numThreads,
                                                     numData)
      .launch_test();
}

int main() {
  // default constructor
  syclcompat::atomic<int> default_constructor;

  INSTANTIATE_ALL_TYPES(atomic_value_type_list, atomic_class_value);
}
