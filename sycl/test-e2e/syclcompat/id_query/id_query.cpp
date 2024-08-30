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
 *  id_query.cpp
 *
 *  Description:
 *    global_id query tests
 **************************************************************************/

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <algorithm>
#include <numeric>

#include <syclcompat/device.hpp>
#include <syclcompat/launch.hpp>

#include "id_query_fixt.hpp"

void global_id_x_query(int *data) {
  data[syclcompat::global_id::x()] = syclcompat::global_id::x();
}
void global_id_y_query(int *data) {
  data[syclcompat::global_id::y()] = syclcompat::global_id::y();
}
void global_id_z_query(int *data) {
  data[syclcompat::global_id::z()] = syclcompat::global_id::z();
}

void test_global_id_query() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  auto checker = [](std::vector<int> input) {
    std::vector<int> expected(input.size());
    std::iota(expected.begin(), expected.end(), 0);
    assert(std::equal(expected.begin(), expected.end(), input.begin()));
  };
  // Check we can query x, y, z components of global_id
  QueryLauncher<global_id_x_query>({4, 1, 1}, {32, 1, 1}).launch_dim3(checker);
  QueryLauncher<global_id_y_query>({1, 4, 1}, {1, 32, 1}).launch_dim3(checker);
  QueryLauncher<global_id_z_query>({1, 1, 4}, {1, 1, 32}).launch_dim3(checker);

  // Check that we can query x component, irrespective of the kernel dimension
  QueryLauncher<global_id_x_query>({4, 1, 1}, {32, 1, 1})
      .launch_ndrange<3>(checker);
  QueryLauncher<global_id_x_query>({4, 1, 1}, {32, 1, 1})
      .launch_ndrange<2>(checker);
  QueryLauncher<global_id_x_query>({4, 1, 1}, {32, 1, 1})
      .launch_ndrange<1>(checker);

  // Check we can query y component for 2D kernel
  QueryLauncher<global_id_y_query>({1, 4, 1}, {1, 32, 1})
      .launch_ndrange<2>(checker);
}

void range_x_query(int *data) {
  data[syclcompat::global_id::x()] = syclcompat::global_range::x() *
                                     syclcompat::work_group_range::x() *
                                     syclcompat::local_range::x();
}

void test_ranges_query() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  // global_range::x() * work_group_range::x() * local_range::x();
  int target = grid.x * threads.x * grid.x * threads.x;

  auto checker = [&](std::vector<int> input) {
    assert(std::all_of(input.begin(), input.end(),
                       [=](int a) { return a == target; }));
  };
  QueryLauncher<range_x_query>(grid, threads).launch_dim3(checker);
}

void wgroup_id_x_query(int *data) {
  data[syclcompat::global_id::x()] = syclcompat::work_group_id::x();
}
void local_id_x_query(int *data) {
  data[syclcompat::global_id::x()] = syclcompat::local_id::x();
}

void test_ids_query() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{4};
  constexpr syclcompat::dim3 threads{32};

  auto wgroup_checker = [&](std::vector<int> input) {
    for (int i = 0; i < input.size(); ++i) {
      assert(input[i] == i / threads.x);
    }
  };
  QueryLauncher<wgroup_id_x_query>(grid, threads).launch_dim3(wgroup_checker);

  auto local_checker = [&](std::vector<int> input) {
    for (int i = 0; i < input.size(); ++i) {
      assert(input[i] == i % threads.x);
    }
  };
  QueryLauncher<local_id_x_query>(grid, threads).launch_dim3(local_checker);
}

int main() {
  test_global_id_query();
  test_ranges_query();
  test_ids_query();

  return 0;
}
