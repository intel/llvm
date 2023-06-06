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
 *  UtilFindFirstSet.cpp
 *
 *  Description:
 *    Find_first_set tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilFindFirstSet.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

void find_first_set_test(int *test_result) {
  int a;
  unsigned long long int lla;
  int result;
  a = 0;
  result = compat::ffs(a);
  if (result != 0) {
    *test_result = 1;
    return;
  }

  a = -2147483648;
  result = compat::ffs(a);
  if (result != 32) {
    *test_result = 1;
    return;
  }

  a = 128;
  result = compat::ffs(a);
  if (result != 8) {
    *test_result = 1;
    return;
  }

  a = 2147483647;
  result = compat::ffs(a);
  if (result != 1) {
    *test_result = 1;
    return;
  }

  lla = -9223372036854775808Ull;
  result = compat::ffs(lla);
  if (result != 64) {
    *test_result = 1;
    return;
  }

  lla = -9223372036854775807Ull;
  result = compat::ffs(lla);
  if (result != 1) {
    *test_result = 1;
    return;
  }

  lla = -9223372034707292160Ull;
  result = compat::ffs(lla);
  if (result != 32) {
    *test_result = 1;
    return;
  }

  lla = 2147483648Ull;
  result = compat::ffs(lla);
  if (result != 32) {
    *test_result = 1;
    return;
  }
}

TEST(Util, find_first_set) {

  compat::device_ext &dev_ct1 = compat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();
  int *test_result, host_test_result = 0;

  test_result = sycl::malloc_shared<int>(sizeof(int), *q_ct1);
  *test_result = 0;

  q_ct1->parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { find_first_set_test(test_result); });

  dev_ct1.queues_wait_and_throw();
  find_first_set_test(&host_test_result);
  EXPECT_EQ(*test_result, 0);
  EXPECT_EQ(host_test_result, 0);
}
