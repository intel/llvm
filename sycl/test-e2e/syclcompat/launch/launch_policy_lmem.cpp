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
 *  launch_policy_lmem.cpp
 *
 *  Description:
 *     launch<F> with policy & use local memory tests
 **************************************************************************/

// XFAIL: arch-intel_gpu_pvc
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14826

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: linux && opencl && (gpu-intel-gen12 || gpu-intel-dg2)
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15275

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/group_barrier.hpp>

#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>
#include <syclcompat/id_query.hpp>

#include "../common.hpp"
#include "launch_fixt.hpp"

namespace compat_exp = syclcompat::experimental;

using compat_exp::launch_policy;
using compat_exp::local_mem_size;

// Kernel functions for testing
// =======================================================================
inline void dynamic_local_mem_empty_kernel(char *a){};

template <typename T>
inline void dynamic_local_mem_basicdt_kernel(T value, char *local_mem){};

template <typename T>
void dynamic_local_mem_typed_kernel(T *data, char *local_mem) {
  constexpr size_t memsize = LaunchTestWithArgs<T>::LOCAL_MEM_SIZE;
  constexpr size_t num_elements = memsize / sizeof(T);
  T *typed_local_mem = reinterpret_cast<T *>(local_mem);

  const int local_id =
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();
  const int group_id =
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id();
  // Only operate in first work-group
  if (group_id == 0) {
    if (local_id < num_elements) {
      typed_local_mem[local_id] = static_cast<T>(local_id);
    }
    syclcompat::wg_barrier();
    if (local_id < num_elements) {
      data[local_id] = typed_local_mem[num_elements - local_id - 1];
    }
  }
};

// =======================================================================

void test_dynamic_mem_no_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_1_, local_mem_size{1}});
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_2_, local_mem_size{1}});
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_3_, local_mem_size{1}});
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.grid_, lt.thread_, local_mem_size{1}});
}

void test_dynamic_mem_no_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_1_, local_mem_size{1}}, lt.q_);
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_2_, local_mem_size{1}}, lt.q_);
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.range_3_, local_mem_size{1}}, lt.q_);
  compat_exp::launch<dynamic_local_mem_empty_kernel>(
      launch_policy{lt.grid_, lt.thread_, local_mem_size{1}}, lt.q_);
}

template <typename T> void test_basic_dt_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  T d_a = T(1);
  LaunchTestWithArgs<T> ltt;

  if (ltt.skip_) // Unsupported aspect
    return;

  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_1_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_2_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_3_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}}, d_a);
}

template <typename T> void test_basic_dt_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  T d_a = T(1);
  LaunchTestWithArgs<T> ltt;

  if (ltt.skip_) // Unsupported aspect
    return;

  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_1_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_2_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.range_3_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_basicdt_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
}

template <typename T> void test_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  T *d_a = (T *)syclcompat::malloc(ltt.memsize_);

  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_1_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_2_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_3_, local_mem_size{ltt.memsize_}}, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}}, d_a);

  syclcompat::wait();
  syclcompat::free(d_a);
}

template <typename T> void test_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  T *d_a = (T *)syclcompat::malloc(ltt.memsize_, ltt.in_order_q_);

  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_1_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_2_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.range_3_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}},
      ltt.in_order_q_, d_a);

  syclcompat::wait(ltt.in_order_q_);
  syclcompat::free(d_a, ltt.in_order_q_);
}

template <typename T> void test_local_mem_usage() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  size_t num_elements = ltt.memsize_ / sizeof(T);

  T *h_a = (T *)syclcompat::malloc_host(ltt.memsize_);
  T *d_a = (T *)syclcompat::malloc(ltt.memsize_);

  // d_a is the kernel output, no memcpy needed
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}}, d_a);

  syclcompat::memcpy(h_a, d_a, ltt.memsize_);
  syclcompat::free(d_a);

  for (int i = 0; i < num_elements; i++) {
    assert(h_a[i] == static_cast<T>(num_elements - i - 1));
  }
  syclcompat::free(h_a);
}

template <typename T> void test_local_mem_usage_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  size_t num_elements = ltt.memsize_ / sizeof(T);
  auto &q = ltt.in_order_q_;

  T *h_a = (T *)syclcompat::malloc_host(ltt.memsize_);
  T *d_a = (T *)syclcompat::malloc(ltt.memsize_, q);

  // d_a is the kernel output, no memcpy needed
  compat_exp::launch<dynamic_local_mem_typed_kernel<T>>(
      launch_policy{ltt.grid_, ltt.thread_, local_mem_size{ltt.memsize_}}, q,
      d_a);

  syclcompat::memcpy(h_a, d_a, ltt.memsize_, q);
  syclcompat::free(d_a, q);

  for (size_t i = 0; i < num_elements; i++) {
    assert(h_a[i] == static_cast<T>(num_elements - i - 1));
  }

  syclcompat::free(h_a);
}

template <typename T> void test_memsize_no_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTest lt;
  T memsize = static_cast<T>(8);

  compat_exp::launch<dynamic_local_mem_empty_kernel>(launch_policy{lt.grid_, lt.thread_,
                                                     local_mem_size(memsize)});
}

template <typename T> void test_memsize_no_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTest lt;
  T memsize = static_cast<T>(8);

  compat_exp::launch<dynamic_local_mem_empty_kernel>(launch_policy{lt.grid_, lt.thread_,
                                                     local_mem_size(memsize)}, lt.q_);
}

int main() {

  test_dynamic_mem_no_arg_launch();
  test_dynamic_mem_no_arg_launch_q();

  INSTANTIATE_ALL_TYPES(value_type_list, test_basic_dt_launch);
  INSTANTIATE_ALL_TYPES(value_type_list, test_basic_dt_launch_q);
  INSTANTIATE_ALL_TYPES(value_type_list, test_arg_launch);
  INSTANTIATE_ALL_TYPES(value_type_list, test_arg_launch_q);
  INSTANTIATE_ALL_TYPES(value_type_list, test_local_mem_usage);
  INSTANTIATE_ALL_TYPES(value_type_list, test_local_mem_usage_q);

  INSTANTIATE_ALL_TYPES(memsize_type_list, test_memsize_no_arg_launch);
  INSTANTIATE_ALL_TYPES(memsize_type_list, test_memsize_no_arg_launch_q);

  return 0;
}
