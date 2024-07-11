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
// https://github.com/intel/llvm/issues/14387
// UNSUPPORTED: gpu-intel-dg2
// RUN: %clangxx -std=c++20 -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <type_traits>

#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/id_query.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/launch_experimental.hpp>
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

  const int id =
      sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  if (id < num_elements) {
    typed_local_mem[id] = static_cast<T>(id);
  }
  sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<1>());
  if (id < num_elements) {
    data[id] = typed_local_mem[num_elements - id - 1];
  }
};

template <typename T>
void reqd_sg_size_kernel(int modifier_val, int num_elements, T *data) {

  const int id = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                     .get_global_linear_id();
  const int sg_size = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                          .get_sub_group()
                          .get_local_linear_range();
  if (id < num_elements) {
    if (id < num_elements - modifier_val) {
      data[id] = static_cast<T>(
          (id + modifier_val - sg_size) < 0 ? 0 : id + modifier_val - sg_size);
    } else {
      data[id] = static_cast<T>(id + modifier_val + sg_size);
    }
  }
}

template <typename T>
void reqd_sg_size_kernel_with_local_memory(int modifier_val, int num_elements,
                                           T *data, char *local_mem) {
  T *typed_local_mem = reinterpret_cast<T *>(local_mem);
  const int id = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                     .get_global_linear_id();
  const int sg_size = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                          .get_sub_group()
                          .get_local_linear_range();

  const int wi_id_in_wg =
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();

  if (id < num_elements - modifier_val) {
    typed_local_mem[wi_id_in_wg] = static_cast<T>(
        (id + modifier_val - sg_size) < 0 ? 0 : id + modifier_val - sg_size);
  } else {
    typed_local_mem[wi_id_in_wg] = static_cast<T>(id + modifier_val + sg_size);
  }

  syclcompat::wg_barrier();

  if (id < num_elements) {
    data[id] = typed_local_mem[wi_id_in_wg];
  }
}

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

template <typename T> void test_reqd_sg_size() {
  namespace syclc_exp = syclcompat::experimental;

  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  int SubgroupSize = 16;
  const int modifier_val = 9;
  const int num_elements = 1024;

  T *h_a = (T *)syclcompat::malloc_host(num_elements * sizeof(T));
  T *d_a = (T *)syclcompat::malloc(num_elements * sizeof(T));
  auto sg_sizes = syclcompat::get_default_queue()
                      .get_device()
                      .get_info<sycl::info::device::sub_group_sizes>();

  if (std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end()) {
    syclc_exp::launch<reqd_sg_size_kernel<T>, 16>(
        ltt.grid_, ltt.thread_, modifier_val, static_cast<int>(num_elements),
        d_a);
  } else {
    SubgroupSize = 32;
    syclc_exp::launch<reqd_sg_size_kernel<T>, 32>(
        ltt.grid_, ltt.thread_, modifier_val, static_cast<int>(num_elements),
        d_a);
  }

  syclcompat::wait_and_throw();
  syclcompat::memcpy<T>(h_a, d_a, num_elements);
  syclcompat::free(d_a);

  for (int i = 0; i < static_cast<int>(num_elements); i++) {
    T result;
    if (i < (static_cast<int>(num_elements) - modifier_val)) {
      result = static_cast<T>((i + modifier_val - SubgroupSize) < 0
                                  ? 0
                                  : (i + modifier_val - SubgroupSize));
    } else {
      result = static_cast<T>(i + modifier_val + SubgroupSize);
    }
    assert(h_a[i] == result);
  }

  syclcompat::free(h_a);
}

template <typename T> void test_reqd_sg_size_q() {
  namespace syclc_exp = syclcompat::experimental;
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;
  int SubgroupSize = 16;
  const int modifier_val = 9;
  auto &q = ltt.in_order_q_;
  const int num_elements = 1024;

  T *h_a = (T *)syclcompat::malloc_host(num_elements * sizeof(T), q);
  T *d_a = (T *)syclcompat::malloc(num_elements * sizeof(T), q);
  sycl::nd_range<3> launch_range(sycl::range<3>(ltt.grid_ * ltt.thread_),
                                 sycl::range<3>(ltt.thread_));
  auto sg_sizes =
      q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end()) {
    syclc_exp::launch<reqd_sg_size_kernel<T>, 16>(
        launch_range, q, modifier_val, static_cast<int>(num_elements), d_a);
  } else {
    SubgroupSize = 32;
    syclc_exp::launch<reqd_sg_size_kernel<T>, 32>(
        launch_range, q, modifier_val, static_cast<int>(num_elements), d_a);
  }

  syclcompat::wait_and_throw();
  syclcompat::memcpy<T>(h_a, d_a, num_elements, q);
  syclcompat::free(d_a, q);

  for (int i = 0; i < static_cast<int>(num_elements); i++) {
    T result;
    if (i < (static_cast<int>(num_elements) - modifier_val)) {
      result = static_cast<T>((i + modifier_val - SubgroupSize) < 0
                                  ? 0
                                  : (i + modifier_val - SubgroupSize));
    } else {
      result = static_cast<T>(i + modifier_val + SubgroupSize);
    }
    assert(h_a[i] == result);
  }
  syclcompat::free(h_a, q);
}

template <typename T> void test_reqd_sg_size_with_local_memory() {
  namespace syclc_exp = syclcompat::experimental;

  std::cout << __PRETTY_FUNCTION__ << std::endl;

  LaunchTestWithArgs<T> ltt;
  if (ltt.skip_) // Unsupported aspect
    return;

  int SubgroupSize = 16;
  const int modifier_val = 9;

  std::size_t local_memory_size =
      ltt.thread_.x * ltt.thread_.y * ltt.thread_.z * sizeof(T);
  auto global_range = ltt.thread_ * ltt.grid_;

  auto num_elements = global_range.x * global_range.y * global_range.z;

  T *h_a = (T *)syclcompat::malloc_host(num_elements * sizeof(T));
  T *d_a = (T *)syclcompat::malloc(num_elements * sizeof(T));

  auto sg_sizes = syclcompat::get_default_queue()
                      .get_device()
                      .get_info<sycl::info::device::sub_group_sizes>();

  if (std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end()) {
    syclc_exp::launch<reqd_sg_size_kernel_with_local_memory<T>, 16>(
        ltt.grid_, ltt.thread_, local_memory_size, modifier_val,
        static_cast<int>(num_elements), d_a);
  } else {
    SubgroupSize = 32;
    syclc_exp::launch<reqd_sg_size_kernel_with_local_memory<T>, 32>(
        ltt.grid_, ltt.thread_, local_memory_size, modifier_val,
        static_cast<int>(num_elements), d_a);
  }

  syclcompat::wait_and_throw();
  syclcompat::memcpy<T>(h_a, d_a, num_elements);

  for (int i = 0; i < static_cast<int>(num_elements); i++) {
    T result;
    if (i < (static_cast<int>(num_elements) - modifier_val)) {
      result = static_cast<T>((i + modifier_val - SubgroupSize) < 0
                                  ? 0
                                  : (i + modifier_val - SubgroupSize));
    } else {
      result = static_cast<T>(i + modifier_val + SubgroupSize);
    }
    assert(h_a[i] == result);
  }
  syclcompat::free(d_a);
  syclcompat::free(h_a);
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

  INSTANTIATE_ALL_TYPES(memsize_type_list, test_reqd_sg_size);
  INSTANTIATE_ALL_TYPES(memsize_type_list, test_reqd_sg_size_q);
  INSTANTIATE_ALL_TYPES(memsize_type_list, test_reqd_sg_size_with_local_memory);

  return 0;
}
