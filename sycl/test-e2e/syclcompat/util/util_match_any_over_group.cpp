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
 *  util_match_any_over_group.cpp
 *
 *  Description:
 *    util_match_any_over_group tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilSelectFromSubGroup.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

constexpr unsigned int NUM_TESTS = 3;
constexpr unsigned int SUBGROUP_SIZE = 32;
constexpr unsigned int DATA_SIZE = NUM_TESTS * SUBGROUP_SIZE;

void test_match_any_over_group() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{DATA_SIZE};

  unsigned int input[DATA_SIZE] = {
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // Subgroup #1
      0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // Subgroup #2
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // Subgroup #3
  };
  unsigned int output[DATA_SIZE];
  unsigned int *d_input = syclcompat::malloc<unsigned int>(DATA_SIZE);
  unsigned int *d_output = syclcompat::malloc<unsigned int>(DATA_SIZE);

  unsigned int member_mask = 0x0FFF;
  unsigned int expected[DATA_SIZE] = {
      0x000F, 0x000F, 0x000F, 0x000F, 0x00F0, 0x00F0, 0x00F0, 0x00F0, //
      0x0F00, 0x0F00, 0x0F00, 0x0F00, 0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      // #1
      0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, //
      0x0F00, 0x0F00, 0x0F00, 0x0F00, 0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      // #2
      0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, //
      0x0FFF, 0x0FFF, 0x0FFF, 0x0FFF, 0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      //
      0,      0,      0,      0,      0,      0,      0,      0,      // #3
  };

  syclcompat::memcpy<unsigned int>(d_input, input, DATA_SIZE);
  sycl::queue q = syclcompat::get_default_queue();
  q.parallel_for(
      sycl::nd_range<1>(grid.size() * threads.size(), threads.size()),
      [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
        auto id = item.get_global_linear_id();
        d_output[id] = syclcompat::match_any_over_sub_group(
            item.get_sub_group(), member_mask, d_input[id]);
      });
  q.wait_and_throw();
  syclcompat::memcpy<unsigned int>(output, d_output, DATA_SIZE);

  for (int i = 0; i < DATA_SIZE; ++i) {
    assert(output[i] == expected[i]);
  }

  syclcompat::free(d_input);
  syclcompat::free(d_output);
}

int main() {
  test_match_any_over_group();

  return 0;
}
