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
 *  atomic_class.cpp
 *
 *  Description:
 *    atomic operations API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ libcu_atomic.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: hip || (windows && level_zero)

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend --cuda-gpu-arch=sm_70 %} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat/atomic.hpp>

#include "../common.hpp"
#include "atomic_fixt.hpp"

constexpr size_t numBlocks = 1;
constexpr size_t numThreads = 1;
constexpr size_t numData = 6;

template <typename T, typename AtomicType>
void atomic_ref_ptr(T *atom_arr_out, T *atom_arr_in) {
  AtomicType a{nullptr};

  a.store(atom_arr_in[0]);

  atom_arr_out[0] = a.load();
  atom_arr_out[1] = a.exchange(atom_arr_in[1]);
  atom_arr_out[2] = a.load();
  a.compare_exchange_weak(atom_arr_out[2], atom_arr_in[2]);
  atom_arr_out[3] = a.load();
  a.compare_exchange_strong(atom_arr_out[3], atom_arr_in[3]);
  atom_arr_out[4] = a.fetch_add(static_cast<std::ptrdiff_t>(1));
  atom_arr_out[5] = a.fetch_sub(static_cast<std::ptrdiff_t>(-1));
}

template <typename T> void atomic_ref_ptr_kernel(T *atom_arr, T *atom_arr_in) {
  atomic_ref_ptr<T, syclcompat::atomic<T>>(atom_arr, atom_arr_in);
}

template <typename T> void atomic_ref_ptr_host(T *atom_arr, T *atom_arr_in) {
  atomic_ref_ptr<T, std::atomic<T>>(atom_arr, atom_arr_in);
}

template <typename T> void test_atomic_class_ptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  AtomicClassPtrTypeLauncher<T>(numBlocks, numThreads, numData)
      .template launch_test<atomic_ref_ptr_kernel<T>, atomic_ref_ptr_host<T>>();
}

template <typename T, typename AtomicType> void atomic_ref_value(T *atom_arr) {
  AtomicType a{static_cast<T>(0)};
  T temp1 = static_cast<T>(3);
  T temp2 = static_cast<T>(4);

  a.store(static_cast<T>(1));

  atom_arr[0] = a.load();
  atom_arr[1] = a.exchange(static_cast<T>(3));
  atom_arr[2] = a.load();
  a.compare_exchange_weak(temp1, static_cast<T>(4));
  atom_arr[3] = a.load();
  a.compare_exchange_strong(temp2, static_cast<T>(8));
  atom_arr[4] = a.fetch_add(static_cast<T>(1));
  atom_arr[5] = a.fetch_sub(static_cast<T>(-1));
}

template <typename T> void atomic_ref_value_kernel(T *atom_arr) {
  atomic_ref_value<T, syclcompat::atomic<T>>(atom_arr);
}

template <typename T> void atomic_ref_value_host(T *atom_arr) {
  // atomic RMW operations for floating point in std is C++20 and may
  // not be implemented
  if constexpr (std::is_integral_v<T>)
    atomic_ref_value<T, std::atomic<T>>(atom_arr);
  else
    atomic_ref_value<T, syclcompat::atomic<T>>(atom_arr);
}

template <typename T> void test_atomic_class_value() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  AtomicClassLauncher<T>(numBlocks, numThreads, numData)
      .template launch_test<atomic_ref_value_kernel<T>,
                            atomic_ref_value_host<T>>();
}

void test_default_constructor() { syclcompat::atomic<int> default_constructor; }

int main() {
  std::vector<sycl::memory_order> supported_memory_orders =
      syclcompat::get_default_queue()
          .get_device()
          .get_info<sycl::info::device::atomic_memory_order_capabilities>();

  if (is_supported(supported_memory_orders, sycl::memory_order::seq_cst)) {
    test_default_constructor();

    INSTANTIATE_ALL_TYPES(atomic_value_type_list, test_atomic_class_value);
    INSTANTIATE_ALL_TYPES(atomic_ptr_type_list, test_atomic_class_ptr);
  }
}
