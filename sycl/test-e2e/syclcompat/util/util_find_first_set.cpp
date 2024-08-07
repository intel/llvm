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
 *  util_find_first_set.cpp
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

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

void find_first_set_test(int *test_result) {
  int a;
  unsigned long long int lla;
  int result;
  a = 0;
  result = syclcompat::ffs(a);
  if (result != 0) {
    *test_result = 1;
    return;
  }

  a = -2147483648;
  result = syclcompat::ffs(a);
  if (result != 32) {
    *test_result = 1;
    return;
  }

  a = 128;
  result = syclcompat::ffs(a);
  if (result != 8) {
    *test_result = 1;
    return;
  }

  a = 2147483647;
  result = syclcompat::ffs(a);
  if (result != 1) {
    *test_result = 1;
    return;
  }

  lla = -9223372036854775808Ull;
  result = syclcompat::ffs(lla);
  if (result != 64) {
    *test_result = 1;
    return;
  }

  lla = -9223372036854775807Ull;
  result = syclcompat::ffs(lla);
  if (result != 1) {
    *test_result = 1;
    return;
  }

  lla = -9223372034707292160Ull;
  result = syclcompat::ffs(lla);
  if (result != 32) {
    *test_result = 1;
    return;
  }

  lla = 2147483648Ull;
  result = syclcompat::ffs(lla);
  if (result != 32) {
    *test_result = 1;
    return;
  }
}

void test_find_first_set() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = *dev_ct1.default_queue();
  int *test_result, host_test_result = 0;

  test_result = sycl::malloc_device<int>(1, q_ct1);
  q_ct1.memcpy(test_result, &host_test_result, sizeof(int)).wait();

  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { find_first_set_test(test_result); });

  dev_ct1.queues_wait_and_throw();
  find_first_set_test(&host_test_result);
  assert(host_test_result == 0);
  q_ct1.memcpy(&host_test_result, test_result, sizeof(int)).wait();
  assert(host_test_result == 0);

  sycl::free(test_result, q_ct1);
}

int main() {
  test_find_first_set();

  return 0;
}
