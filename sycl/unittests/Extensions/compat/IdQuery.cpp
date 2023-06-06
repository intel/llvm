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
 *  SYCL compatibility API
 *
 *  IdQuery.cpp
 *
 *  Description:
 *    global_id query tests
 **************************************************************************/

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// Class to launch a kernel and run a lambda on output data
template <auto F> class QueryLauncher {
protected:
  compat::dim3 grid_;
  compat::dim3 threads_;
  size_t size_;
  int *data_;
  std::vector<int> host_data_;
  using CheckLambda = std::function<void(std::vector<int>)>;

public:
  QueryLauncher(compat::dim3 grid, compat::dim3 threads)
      : grid_{grid}, threads_{threads}, size_{grid_.size() * threads_.size()},
        host_data_(size_) {
    data_ = (int *)compat::malloc(size_ * sizeof(int));
    compat::memset(data_, 0, size_ * sizeof(int));
  };
  ~QueryLauncher() { compat::free(data_); }
  template <typename... Args>
  void launch_dim3(CheckLambda checker, Args... args) {
    compat::launch<F>(grid_, threads_, data_, args...);
    compat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    compat::wait();
    checker(host_data_);
  }
  template <int Dim, typename... Args>
  void launch_ndrange(CheckLambda checker, Args... args) {
    sycl::nd_range<Dim> range = {grid_ * threads_, grid_};
    compat::launch<F>(range, data_, args...);
    compat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    compat::wait();
    checker(host_data_);
  }
};

void global_id_x_query(int *data) {
  data[compat::global_id::x()] = compat::global_id::x();
}
void global_id_y_query(int *data) {
  data[compat::global_id::y()] = compat::global_id::y();
}
void global_id_z_query(int *data) {
  data[compat::global_id::z()] = compat::global_id::z();
}

TEST(queries, global_id_query) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  auto checker = [](std::vector<int> input) {
    std::vector<int> expected(input.size());
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), input.begin()));
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
  data[compat::global_id::x()] = compat::global_range::x() *
                                 compat::work_group_range::x() *
                                 compat::local_range::x();
}

TEST(queries, ranges_query) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  // global_range::x() * work_group_range::x() * local_range::x();
  int target = grid.x * threads.x * grid.x * threads.x;

  auto checker = [&](std::vector<int> input) {
    EXPECT_TRUE(std::all_of(input.begin(), input.end(),
                            [=](int a) { return a == target; }));
  };
  QueryLauncher<range_x_query>(grid, threads).launch_dim3(checker);
}

void wgroup_id_x_query(int *data) {
  data[compat::global_id::x()] = compat::work_group_id::x();
}
void local_id_x_query(int *data) {
  data[compat::global_id::x()] = compat::local_id::x();
}

TEST(queries, ids_query) {
  constexpr compat::dim3 grid{4};
  constexpr compat::dim3 threads{32};

  auto wgroup_checker = [&](std::vector<int> input) {
    for (int i = 0; i < input.size(); ++i) {
      if (input[i] != i / threads.x)
        FAIL();
    }
  };
  QueryLauncher<wgroup_id_x_query>(grid, threads).launch_dim3(wgroup_checker);

  auto local_checker = [&](std::vector<int> input) {
    for (int i = 0; i < input.size(); ++i) {
      if (input[i] != i % threads.x)
        FAIL();
    }
  };
  QueryLauncher<local_id_x_query>(grid, threads).launch_dim3(local_checker);
}
