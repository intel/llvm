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
 *  launch.cpp
 *
 *  Description:
 *     launch<F> and launch<F> with dinamyc local memory tests
 **************************************************************************/

// RUN: %clangxx -std=c++20 -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <type_traits>

#include <sycl/sycl.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "launch_fixt.hpp"

// Dummy kernel functions for testing
inline void empty_kernel(){};
inline void int_kernel(int a){};
inline void int_ptr_kernel(int *a){};
inline void dynamic_local_mem_empty_kernel(char *a){};

template <typename T>
inline void dynamic_local_mem_basicdt_kernel(T value, char *local_mem){};

template <typename T>
void dynamic_local_mem_typed_kernel(T *data, char *local_mem) {
  constexpr size_t memsize = LaunchTestWithArgs<T>::LOCAL_MEM_SIZE;
  constexpr size_t num_elements = memsize / sizeof(T);
  T *typed_local_mem = reinterpret_cast<T *>(local_mem);

  const int id = sycl::ext::oneapi::experimental::this_item<1>();
  if (id < num_elements) {
    typed_local_mem[id] = static_cast<T>(id);
  }
  sycl::group_barrier(sycl::ext::oneapi::experimental::this_group<1>());
  if (id < num_elements) {
    data[id] = typed_local_mem[num_elements - id - 1];
  }
};

template <int Dim>
void compute_nd_range_3d(RangeParams<Dim> range_param, std::string test_name) {
  std::cout << __PRETTY_FUNCTION__ << " " << test_name << std::endl;

  try {
    auto g_out = syclcompat::compute_nd_range(range_param.global_range_in_,
                                              range_param.local_range_in_);
    sycl::nd_range<Dim> x_out = {range_param.expect_global_range_out_,
                                 range_param.local_range_in_};
    if (range_param.shouldPass_) {
      assert(g_out == x_out);
    } else {
      assert(false); // Trigger failure, expected std::invalid_argument
    }
  } catch (std::invalid_argument const &err) {
    if (range_param.shouldPass_) {
      assert(false); // Trigger failure, unexpected std::invalid_argument
    }
  }
}

void test_launch_compute_nd_range_3d() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  compute_nd_range_3d(RangeParams<3>{{11, 1, 1}, {2, 1, 1}, {12, 1, 1}, true},
                      "Round up");
  compute_nd_range_3d(
      RangeParams<3>{{320, 1, 1}, {32, 1, 1}, {320, 1, 1}, true}, "Even size");
  compute_nd_range_3d(
      RangeParams<3>{{32, 193, 1}, {16, 32, 1}, {32, 224, 1}, true},
      "Round up 2");
  compute_nd_range_3d(RangeParams<3>{{10, 0, 0}, {1, 0, 0}, {10, 0, 0}, false},
                      "zero size");
  compute_nd_range_3d(
      RangeParams<3>{{0, 10, 10}, {0, 10, 10}, {0, 10, 10}, false},
      "zero size 2");
  compute_nd_range_3d(RangeParams<3>{{2, 1, 1}, {32, 1, 1}, {32, 1, 1}, false},
                      "local > global");
  compute_nd_range_3d(RangeParams<3>{{1, 2, 1}, {1, 32, 1}, {1, 32, 1}, false},
                      "local > global 2");
  compute_nd_range_3d(RangeParams<3>{{1, 1, 2}, {1, 1, 32}, {1, 1, 32}, false},
                      "local > global 3");
}

void test_no_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  syclcompat::launch<empty_kernel>(lt.range_1_);
  syclcompat::launch<empty_kernel>(lt.range_2_);
  syclcompat::launch<empty_kernel>(lt.range_3_);
  syclcompat::launch<empty_kernel>(lt.grid_, lt.thread_);

  syclcompat::launch<empty_kernel>(lt.range_1_, lt.q_);
  syclcompat::launch<empty_kernel>(lt.range_2_, lt.q_);
  syclcompat::launch<empty_kernel>(lt.range_3_, lt.q_);
  syclcompat::launch<empty_kernel>(lt.grid_, lt.thread_, lt.q_);
}

void test_one_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  int my_int;

  syclcompat::launch<int_kernel>(lt.range_1_, my_int);
  syclcompat::launch<int_kernel>(lt.range_2_, my_int);
  syclcompat::launch<int_kernel>(lt.range_3_, my_int);
  syclcompat::launch<int_kernel>(lt.grid_, lt.thread_, my_int);

  syclcompat::launch<int_kernel>(lt.range_1_, lt.q_, my_int);
  syclcompat::launch<int_kernel>(lt.range_2_, lt.q_, my_int);
  syclcompat::launch<int_kernel>(lt.range_3_, lt.q_, my_int);
  syclcompat::launch<int_kernel>(lt.grid_, lt.thread_, lt.q_, my_int);
}

void test_ptr_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  int *int_ptr;

  syclcompat::launch<int_ptr_kernel>(lt.range_1_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.range_2_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.range_3_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.grid_, lt.thread_, int_ptr);

  syclcompat::launch<int_ptr_kernel>(lt.range_1_, lt.q_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.range_2_, lt.q_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.range_3_, lt.q_, int_ptr);
  syclcompat::launch<int_ptr_kernel>(lt.grid_, lt.thread_, lt.q_, int_ptr);
}

void test_dynamic_mem_no_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_1_, 1);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_2_, 1);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_3_, 1);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.grid_, lt.thread_, 1);
}

void test_dynamic_mem_no_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  LaunchTest lt;

  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_1_, 1, lt.q_);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_2_, 1, lt.q_);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.range_3_, 1, lt.q_);
  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.grid_, lt.thread_, 1,
                                                     lt.q_);
}

template <typename T> void test_basic_dt_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  T d_a = T(1);
  LaunchTestWithArgs<T> ltt;

  if (ltt.skip_) // Unsupported aspect
    return;

  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(ltt.range_1_,
                                                          ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(ltt.range_2_,
                                                          ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(ltt.range_3_,
                                                          ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(
      ltt.grid_, ltt.thread_, ltt.memsize_, d_a);
}

template <typename T> void test_basic_dt_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  T d_a = T(1);
  LaunchTestWithArgs<T> ltt;

  if (ltt.skip_) // Unsupported aspect
    return;

  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(
      ltt.range_1_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(
      ltt.range_2_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(
      ltt.range_3_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_basicdt_kernel<T>>(
      ltt.grid_, ltt.thread_, ltt.memsize_, ltt.in_order_q_, d_a);
}

template <typename T> void test_arg_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  T *d_a = (T *)syclcompat::malloc(ltt.memsize_);

  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.range_1_,
                                                        ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.range_2_,
                                                        ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.range_3_,
                                                        ltt.memsize_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.grid_, ltt.thread_,
                                                        ltt.memsize_, d_a);

  syclcompat::free(d_a);
}

template <typename T> void test_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  T *d_a = (T *)syclcompat::malloc(ltt.memsize_, ltt.in_order_q_);

  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(
      ltt.range_1_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(
      ltt.range_2_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(
      ltt.range_3_, ltt.memsize_, ltt.in_order_q_, d_a);
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(
      ltt.grid_, ltt.thread_, ltt.memsize_, ltt.in_order_q_, d_a);

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
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.grid_, ltt.thread_,
                                                        ltt.memsize_, d_a);

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
  syclcompat::launch<dynamic_local_mem_typed_kernel<T>>(ltt.grid_, ltt.thread_,
                                                        ltt.memsize_, q, d_a);

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

  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.grid_, lt.thread_,
                                                     memsize);
}

template <typename T> void test_memsize_no_arg_launch_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTest lt;
  T memsize = static_cast<T>(8);

  syclcompat::launch<dynamic_local_mem_empty_kernel>(lt.grid_, lt.thread_,
                                                     memsize, lt.q_);
}

int main() {
  test_launch_compute_nd_range_3d();
  test_no_arg_launch();
  test_one_arg_launch();
  test_ptr_arg_launch();

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
