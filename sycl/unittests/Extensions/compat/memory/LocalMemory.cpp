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
 *  LocalMemory.cpp
 *
 *  Description:
 *    launch<F> tests with static local memory
 **************************************************************************/

#include <gtest/gtest.h>
#include <numeric>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

template <auto F> class LocalMemTest {
public:
  LocalMemTest(compat::dim3 grid, compat::dim3 threads)
      : grid_{grid}, threads_{threads}, size_{grid_.size() * threads_.size()},
        host_data_(size_) {
    data_ = (int *)compat::malloc(size_ * sizeof(int));
  };
  ~LocalMemTest() { compat::free(data_); };

  template <typename Lambda, typename... Args>
  void launch_test(Lambda checker, Args... args) {
    compat::launch<F>(grid_, threads_, data_, args...);
    compat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    checker(host_data_);
  }

private:
  compat::dim3 grid_;
  compat::dim3 threads_;
  size_t size_;
  sycl::queue q_;
  int *data_;
  std::vector<int> host_data_;
  using CheckLambda = std::function<void(std::vector<int>)>;
};

// 1D test
// Write id to linear block, then reverse order
template <int BLOCK_SIZE> void local_mem_1d(int *d_A) {
  int *as = compat::local_mem<int[BLOCK_SIZE]>();
  int id = compat::local_id::x();
  as[id] = id;
  compat::wg_barrier();
  int val = as[BLOCK_SIZE - id - 1];
  d_A[compat::global_id::x()] = val;
}

TEST(LocalMemTest, local_1d) {
  auto checker = [](std::vector<int> input) {
    std::vector<int> expected(input.size());
    std::iota(expected.rbegin(), expected.rend(), 0);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), input.begin()));
  };
  LocalMemTest<local_mem_1d<32>>(1, 32).launch_test(checker);
}

// 2D test
// Write id to 2D block, then reverse order
template <int BLOCK_SIZE> void local_mem_2d(int *d_A) {
  auto as = compat::local_mem<int[BLOCK_SIZE][BLOCK_SIZE]>();
  int id_x = compat::local_id::x();
  int id_y = compat::local_id::y();
  as[id_y][id_x] = id_x * BLOCK_SIZE + id_y;
  compat::wg_barrier();
  int val = as[BLOCK_SIZE - id_y - 1][BLOCK_SIZE - id_x - 1];
  d_A[compat::global_id::y() * BLOCK_SIZE + compat::global_id::x()] = val;
}

TEST(LocalMemTest, local_2d) {
  constexpr int TILE_SIZE = 16;
  auto checker = [](std::vector<int> input) {
    for (int y = 0; y < TILE_SIZE; ++y) {
      for (int x = 0; x < TILE_SIZE; ++x) {
        int linear_id = y * TILE_SIZE + x;
        int expected = ((TILE_SIZE - x - 1) * TILE_SIZE) + (TILE_SIZE - y - 1);
        EXPECT_EQ(input[linear_id], expected);
      }
    }
  };
  LocalMemTest<local_mem_2d<TILE_SIZE>>({1, 1}, {TILE_SIZE, TILE_SIZE})
      .launch_test(checker);
}

// 3D test
// Write id to 3D block, then reverse order
template <int BLOCK_SIZE> void local_mem_3d(int *d_A) {
  auto as = compat::local_mem<int[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE]>();
  int id_x = compat::local_id::x();
  int id_y = compat::local_id::y();
  int id_z = compat::local_id::z();
  as[id_z][id_y][id_x] =
      (id_x * (BLOCK_SIZE * BLOCK_SIZE)) + (id_y * BLOCK_SIZE) + id_z;
  compat::wg_barrier();
  int val =
      as[BLOCK_SIZE - id_z - 1][BLOCK_SIZE - id_y - 1][BLOCK_SIZE - id_x - 1];
  d_A[compat::global_id::z() * BLOCK_SIZE * BLOCK_SIZE +
      compat::global_id::y() * BLOCK_SIZE + compat::global_id::x()] = val;
}

TEST(LocalMemTest, local_3d) {
  constexpr int TILE_SIZE = 4;
  auto checker = [](std::vector<int> input) {
    for (int z = 0; z < TILE_SIZE; ++z) {
      for (int y = 0; y < TILE_SIZE; ++y) {
        for (int x = 0; x < TILE_SIZE; ++x) {
          int linear_id = z * TILE_SIZE * TILE_SIZE + y * TILE_SIZE + x;
          int expected = ((TILE_SIZE - x - 1) * TILE_SIZE * TILE_SIZE) +
                         ((TILE_SIZE - y - 1) * TILE_SIZE) +
                         (TILE_SIZE - z - 1);
          EXPECT_EQ(input[linear_id], expected);
        }
      }
    }
  };
  LocalMemTest<local_mem_3d<TILE_SIZE>>({1, 1, 1},
                                        {TILE_SIZE, TILE_SIZE, TILE_SIZE})
      .launch_test(checker);
}
