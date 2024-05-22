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
 *  util_occupancy_calculation.cpp
 *
 *  Description:
 *    max_potential_wg and max_active_wg tests
 **************************************************************************/

// The original source was under the license below:
// ====--- util_calculate_max_active_wg_per_xecore.cpp ----- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

// REQUIRES: gpu
// REQUIRES: level_zero || opencl

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/util.hpp>

// These tests only check the API, not the functionality itself.
void test_calculate_max_active_wg_per_xecore() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  int sg_size = 32;
  bool used_barrier = true;
  bool used_large_grf = true;
  syclcompat::experimental::calculate_max_active_wg_per_xecore(
      &num_blocks, block_size, dynamic_shared_memory_size, sg_size,
      used_barrier, used_large_grf);
}

void test_calculate_max_potential_wg() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  int sg_size = 32;
  bool used_barrier = true;
  bool used_large_grf = true;

  int block_size_limit = 0;
  syclcompat::experimental::calculate_max_potential_wg(
      &num_blocks, &block_size, block_size_limit, dynamic_shared_memory_size,
      sg_size, used_barrier, used_large_grf);
}

int main() {
  test_calculate_max_active_wg_per_xecore();
  test_calculate_max_potential_wg();

  return 0;
}
