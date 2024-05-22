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
 *  util_logical_group.cpp
 *
 *  Description:
 *    logical_group tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilLogicalGroup.cpp -------------------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// REQUIRES: sg-32
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <cstdio>

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

// work-item:
// 0 ... 7, 8 ... 15, 16 ... 23, 24 ... 31, 32 ... 39, 40 ... 47, 48 ... 51
// -------  --------  ---------  ---------  ---------  ---------  ---------
// 0        1         2          3          4          5          6

void kernel(unsigned int *result, sycl::nd_item<3> item_ct1) {
  auto ttb = item_ct1.get_group();
  syclcompat::experimental::logical_group tbt =
      syclcompat::experimental::logical_group(item_ct1, item_ct1.get_group(),
                                              8);

  if (item_ct1.get_local_id(2) == 50) {
    result[0] = tbt.get_local_linear_range();
    result[1] = tbt.get_local_linear_id();
    result[2] = tbt.get_group_linear_range();
    result[3] = tbt.get_group_linear_id();
  }
}

void test_logical_group() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = *dev_ct1.default_queue();
  unsigned int result_host[4];
  unsigned int *result_device;
  result_device = sycl::malloc_device<unsigned int>(4, q_ct1);
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 52), sycl::range<3>(1, 1, 52)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        kernel(result_device, item_ct1);
      });
  q_ct1.memcpy(result_host, result_device, sizeof(unsigned int) * 4).wait();
  sycl::free(result_device, q_ct1);

  assert(result_host[0] == 4);
  assert(result_host[1] == 2);
  assert(result_host[2] == 7);
  assert(result_host[3] == 6);
}

int main() {
  test_logical_group();

  return 0;
}
