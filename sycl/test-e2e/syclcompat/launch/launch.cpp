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
// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14387
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/id_query.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

#include "launch_fixt.hpp"

// Dummy kernel functions for testing
inline void empty_kernel(){};
inline void int_kernel(int a){};
inline void int_ptr_kernel(int *a){};

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

int main() {
  test_launch_compute_nd_range_3d();
  test_no_arg_launch();
  test_one_arg_launch();
  test_ptr_arg_launch();

  return 0;
}
