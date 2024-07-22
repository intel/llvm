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
 *  local_memory.cpp
 *
 *  Description:
 *    launch<F> tests with static local memory
 **************************************************************************/

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <numeric>

#include <syclcompat/device.hpp>
#include <syclcompat/id_query.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

#include "memory_fixt.hpp"

// 1D test
// Write id to linear block, then reverse order
template <int BLOCK_SIZE> void local_mem_1d(int *d_A) {
  int *as = syclcompat::local_mem<int[BLOCK_SIZE]>();
  int id = syclcompat::local_id::x();
  as[id] = id;
  syclcompat::wg_barrier();
  int val = as[BLOCK_SIZE - id - 1];
  d_A[syclcompat::global_id::x()] = val;
}

void test_local_1d() {
  auto checker = [](std::vector<int> input) {
    std::vector<int> expected(input.size());
    std::iota(expected.rbegin(), expected.rend(), 0);
    assert(std::equal(expected.begin(), expected.end(), input.begin()));
  };
  LocalMemTest<local_mem_1d<32>>(1, 32).launch_test(checker);
}

// 2D test
// Write id to 2D block, then reverse order
template <int BLOCK_SIZE> void local_mem_2d(int *d_A) {
  auto as = syclcompat::local_mem<int[BLOCK_SIZE][BLOCK_SIZE]>();
  int id_x = syclcompat::local_id::x();
  int id_y = syclcompat::local_id::y();
  as[id_y][id_x] = id_x * BLOCK_SIZE + id_y;
  syclcompat::wg_barrier();
  int val = as[BLOCK_SIZE - id_y - 1][BLOCK_SIZE - id_x - 1];
  d_A[syclcompat::global_id::y() * BLOCK_SIZE + syclcompat::global_id::x()] =
      val;
}

void test_local_2d() {
  constexpr int TILE_SIZE = 16;
  auto checker = [](std::vector<int> input) {
    for (int y = 0; y < TILE_SIZE; ++y) {
      for (int x = 0; x < TILE_SIZE; ++x) {
        int linear_id = y * TILE_SIZE + x;
        int expected = ((TILE_SIZE - x - 1) * TILE_SIZE) + (TILE_SIZE - y - 1);
        assert(input[linear_id] == expected);
      }
    }
  };
  LocalMemTest<local_mem_2d<TILE_SIZE>>({1, 1}, {TILE_SIZE, TILE_SIZE})
      .launch_test(checker);
}

// 3D test
// Write id to 3D block, then reverse order
template <int BLOCK_SIZE> void local_mem_3d(int *d_A) {
  auto as = syclcompat::local_mem<int[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE]>();
  int id_x = syclcompat::local_id::x();
  int id_y = syclcompat::local_id::y();
  int id_z = syclcompat::local_id::z();
  as[id_z][id_y][id_x] =
      (id_x * (BLOCK_SIZE * BLOCK_SIZE)) + (id_y * BLOCK_SIZE) + id_z;
  syclcompat::wg_barrier();
  int val =
      as[BLOCK_SIZE - id_z - 1][BLOCK_SIZE - id_y - 1][BLOCK_SIZE - id_x - 1];
  d_A[syclcompat::global_id::z() * BLOCK_SIZE * BLOCK_SIZE +
      syclcompat::global_id::y() * BLOCK_SIZE + syclcompat::global_id::x()] =
      val;
}

void test_local_3d() {
  constexpr int TILE_SIZE = 4;
  auto checker = [](std::vector<int> input) {
    for (int z = 0; z < TILE_SIZE; ++z) {
      for (int y = 0; y < TILE_SIZE; ++y) {
        for (int x = 0; x < TILE_SIZE; ++x) {
          int linear_id = z * TILE_SIZE * TILE_SIZE + y * TILE_SIZE + x;
          int expected = ((TILE_SIZE - x - 1) * TILE_SIZE * TILE_SIZE) +
                         ((TILE_SIZE - y - 1) * TILE_SIZE) +
                         (TILE_SIZE - z - 1);
          assert(input[linear_id] == expected);
        }
      }
    }
  };
  LocalMemTest<local_mem_3d<TILE_SIZE>>({1, 1, 1},
                                        {TILE_SIZE, TILE_SIZE, TILE_SIZE})
      .launch_test(checker);
}

int main() {
  test_local_1d();
  test_local_2d();
  test_local_3d();

  return 0;
}
