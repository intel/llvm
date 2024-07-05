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
 *  launch_config.cpp
 *
 *  Description:
 *     launch<F> with config tests
 **************************************************************************/

// RUN: %clangxx -std=c++20 -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include "sycl/ext/oneapi/experimental/root_group.hpp"
#include <sycl/ext/oneapi/properties/properties.hpp>
#include "sycl/ext/oneapi/kernel_properties/properties.hpp"
#include "sycl/ext/intel/experimental/kernel_execution_properties.hpp"
#include "syclcompat/device.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>

//#include <syclcompat/launch.hpp>
#include <syclcompat/launch_strategy.hpp>
#include <syclcompat/memory.hpp>


#include "launch_fixt.hpp"

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;
namespace sycl_intel_exp = sycl::ext::intel::experimental;

// Dummy kernel functions for testing
// =======================================================================

static constexpr int LOCAL_MEM_SIZE = 1024;

inline void empty_kernel(){};
inline void int_kernel(int a){};
inline void int_ptr_kernel(int *a){};

inline void dynamic_local_mem_empty_kernel(char *a){};

template <typename T>
inline void dynamic_local_mem_basicdt_kernel(T value, char *local_mem){};

template <typename T>
void dynamic_local_mem_typed_kernel(T *data, char *local_mem) {
  constexpr size_t num_elements = LOCAL_MEM_SIZE / sizeof(T);
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
// =======================================================================

int test_basic_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  // TODO: ensure we have a combo of, launch_property, kernel_property,
  // runtime_value, param/paramless
  // local memory usage, reducer??

  // A runtime kernel property for the

  sycl_intel_exp::cache_config my_cache_config{sycl_intel_exp::large_slm}; // constructed at runtime

  compat_exp::launch_strategy my_config(
      sycl::nd_range<1>{{32}, {32}},
      sycl_exp::properties{sycl_exp::sub_group_size<32>,
                           sycl_exp::use_root_sync, my_cache_config},
      sycl_exp::properties{}, 0);

  //TODO: deal with initializer list ctor here
  // compat_exp::launch_strategy my_init_config{
  //     sycl::nd_range<1>{{32}, {32}},
  //     sycl_exp::properties{sycl_exp::sub_group_size<32>,
  //                          sycl_exp::use_root_sync, my_cache_config},
  //     sycl_exp::properties{}, local_mem_size};

  sycl::queue q = syclcompat::get_default_queue();

  int dummy_int{1};

  compat_exp::launch<empty_kernel>(my_config);
  compat_exp::launch<int_kernel>(my_config, dummy_int);

  compat_exp::launch<empty_kernel>(my_config, q);
  compat_exp::launch<int_kernel>(my_config, q, dummy_int);

  return 0;
}

int test_lmem_launch() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using T = int;
  // A property constructed at runtime:
  sycl_intel_exp::cache_config my_cache_config{sycl_intel_exp::large_slm}; 

  int local_mem_size = LOCAL_MEM_SIZE; // rt value

  size_t num_elements = local_mem_size / sizeof(T);
  T *h_a = (T *)syclcompat::malloc_host(local_mem_size);
  T *d_a = (T *)syclcompat::malloc(local_mem_size);

  compat_exp::launch_strategy my_config(
      sycl::nd_range<1>{{256}, {256}},
      sycl_exp::properties{sycl_exp::sub_group_size<32>, sycl_exp::use_root_sync, my_cache_config},
      sycl_exp::properties{}, local_mem_size);

  compat_exp::launch<dynamic_local_mem_empty_kernel>(my_config).wait();
  std::cout << "Launched 1 succesfully" << std::endl;
  compat_exp::launch<dynamic_local_mem_typed_kernel<int>>(
      my_config, d_a).wait();
  std::cout << "Launched 2 succesfully" << std::endl;


  syclcompat::memcpy(h_a, d_a, local_mem_size);
  syclcompat::free(d_a);

  for (int i = 0; i < num_elements; i++) {
    assert(h_a[i] == static_cast<T>(num_elements - i - 1));
  }
  syclcompat::free(h_a);
  return 0;
}

int test_dim3_launch_strategy() {

  compat_exp::launch_strategy my_range_config(
      syclcompat::dim3{32}, sycl_exp::properties{}, sycl_exp::properties{}, 0);

  static_assert(
      std::is_same_v<decltype(my_range_config)::RangeT, sycl::range<3>>);

  compat_exp::launch_strategy my_nd_range_config(
      syclcompat::dim3{32}, syclcompat::dim3{32}, sycl_exp::properties{},
      sycl_exp::properties{}, 0);

  static_assert(
      std::is_same_v<decltype(my_nd_range_config)::RangeT, sycl::nd_range<3>>);


  compat_exp::launch<empty_kernel>(my_range_config).wait();
  std::cout << "Launched 1 succesfully" << std::endl;
  compat_exp::launch<empty_kernel>(my_nd_range_config).wait();
  std::cout << "Launched 2 succesfully" << std::endl;

  return 0;
}

// TODO: negative testing for traits/templates

int main() {
  // TODO: check return values!
  test_basic_launch();
  test_lmem_launch();
}
