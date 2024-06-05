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
 *  usm_allocations.cpp
 *
 *  Description:
 *    USM allocation tests
 **************************************************************************/

// REQUIRES: aspect-usm_shared_allocations
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <numeric>

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "../common.hpp"
#include "memory_common.hpp"
#include "memory_fixt.hpp"

template <typename T> void test_shared() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  USMTest<T> usm_fixture;

  if (usm_fixture.skip)
    return; // Skip unsupported

  usm_fixture.data = syclcompat::malloc_shared<T>(usm_fixture.size_);
  usm_fixture.launch_kernel();
  usm_fixture.check_result();
  syclcompat::free(usm_fixture.data);
}

void test_non_templated_shared() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  USMTest<int> usm_fixture;

  usm_fixture.data = static_cast<int *>(
      syclcompat::malloc_shared(usm_fixture.size_ * sizeof(int)));
  usm_fixture.launch_kernel();
  usm_fixture.check_result();
  syclcompat::free(usm_fixture.data);
}

// Test deduce direction
void test_deduce_shared() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using namespace syclcompat::experimental;
  auto default_queue = syclcompat::get_default_queue();

  int *h_ptr = (int *)syclcompat::malloc_host(sizeof(int));
  int *sys_ptr = (int *)std::malloc(sizeof(int));
  int *d_ptr = (int *)syclcompat::malloc(sizeof(int));
  int *s_ptr = (int *)syclcompat::malloc_shared(sizeof(int));

  // * to host
  assert(syclcompat::detail::deduce_memcpy_direction(default_queue, h_ptr,
                                                     s_ptr) ==
         memcpy_direction::device_to_device);

  // * to sys
  assert(syclcompat::detail::deduce_memcpy_direction(
             default_queue, sys_ptr, s_ptr) == memcpy_direction::host_to_host);

  // * to dev
  assert(syclcompat::detail::deduce_memcpy_direction(default_queue, d_ptr,
                                                     s_ptr) ==
         memcpy_direction::device_to_device);

  // * to shared
  assert(syclcompat::detail::deduce_memcpy_direction(default_queue, s_ptr,
                                                     h_ptr) ==
         memcpy_direction::device_to_device);
  assert(syclcompat::detail::deduce_memcpy_direction(
             default_queue, s_ptr, sys_ptr) == memcpy_direction::host_to_host);
  assert(syclcompat::detail::deduce_memcpy_direction(default_queue, s_ptr,
                                                     d_ptr) ==
         memcpy_direction::device_to_device);
  assert(syclcompat::detail::deduce_memcpy_direction(default_queue, s_ptr,
                                                     s_ptr) ==
         memcpy_direction::device_to_device);

  syclcompat::free(s_ptr);
  std::free(sys_ptr);
  syclcompat::free(h_ptr);
  syclcompat::free(d_ptr);
}

int main() {
  INSTANTIATE_ALL_TYPES(value_type_list, test_shared);
  test_non_templated_shared();
  test_deduce_shared();

  return 0;
}
