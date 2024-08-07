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
 *  util_match_all_over_group.cpp
 *
 *  Description:
 *    util_match_all_over_group tests
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

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

constexpr unsigned int NUM_TESTS = 3;
constexpr unsigned int SUBGROUP_SIZE = 32;
constexpr unsigned int DATA_SIZE = NUM_TESTS * SUBGROUP_SIZE;

void test_match_all_over_group() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  constexpr syclcompat::dim3 grid{1};
  constexpr syclcompat::dim3 threads{SUBGROUP_SIZE};

  unsigned int input[DATA_SIZE] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, // #1
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, // #2
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, // #3
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  unsigned int output[DATA_SIZE];
  int pred[DATA_SIZE];
  unsigned int *d_input = syclcompat::malloc<unsigned int>(DATA_SIZE);
  unsigned int *d_output = syclcompat::malloc<unsigned int>(DATA_SIZE);
  int *d_pred = syclcompat::malloc<int>(DATA_SIZE);

  unsigned int member_mask = 0x00FF;
  unsigned int expected[DATA_SIZE] = {
      0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, // #1
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
      0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, // #2
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0, // #3
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,
  };
  unsigned int expected_pred[DATA_SIZE] = {
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // #1
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // #2
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // #3
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  syclcompat::memcpy<unsigned int>(d_input, input, DATA_SIZE);
  syclcompat::memset(d_output, 0, DATA_SIZE * sizeof(unsigned int));
  syclcompat::memset(d_pred, 1, DATA_SIZE * sizeof(int));

  sycl::queue q = syclcompat::get_default_queue();
  q.parallel_for(
      sycl::nd_range<1>(threads.size(), threads.size()),
      [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
        for (auto id = item.get_global_linear_id(); id < DATA_SIZE;
             id += SUBGROUP_SIZE)
          d_output[id] = syclcompat::match_all_over_sub_group(
              item.get_sub_group(), member_mask, d_input[id], &d_pred[id]);
      });
  q.wait_and_throw();
  syclcompat::memcpy<unsigned int>(output, d_output, DATA_SIZE);
  syclcompat::memcpy<int>(pred, d_pred, DATA_SIZE);

  for (int i = 0; i < DATA_SIZE; ++i) {
    assert(output[i] == expected[i]);
    assert(pred[i] == expected_pred[i]);
  }

  syclcompat::free(d_input);
  syclcompat::free(d_output);
  syclcompat::free(d_pred);
}

int main() {
  test_match_all_over_group();

  return 0;
}
