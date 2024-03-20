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
 *  memory_async.cpp
 *
 *  Description:
 *    Asynchronous memory operations event dependency tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_async.cpp------------------- -*- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

// Tests for the sycl::events returned from syclcompat::*Async API calls

#include <stdio.h>

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "memory_fixt.hpp"

// free_async is a host task, so we are really testing the event dependency here
void test_free_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  float *d_D = (float *)syclcompat::malloc(sizeof(float));
  sycl::event kernel_ev = atest.launch_kernel();
  sycl::event free_ev = syclcompat::free_async({d_D}, {kernel_ev});

  atest.check_events(kernel_ev, free_ev);
}

// The following tests are simply testing (as best possible) that
// the sycl::event returned from *Async really corresponds to the task
// We don't check that the memory operation does what it's supposed to,
// this is tested elsewhere.
void test_memcpy_async1() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  sycl::event memcpy_ev = syclcompat::memcpy_async(atest.d_A_, atest.d_C_,
                                                   sizeof(float) * atest.size_);
  sycl::event host_ev = atest.launch_host_task({memcpy_ev});

  atest.check_events(memcpy_ev, host_ev);
}

void test_memcpy_async2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  sycl::event memcpy_ev =
      syclcompat::memcpy_async(atest.d_A_, 32, atest.d_C_, 32, 32, 4);
  sycl::event host_ev = atest.launch_host_task({memcpy_ev});

  atest.check_events(memcpy_ev, host_ev);
}

void test_memcpy_async3() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  assert(width * height * depth <= atest.size_);

  syclcompat::pitched_data d_A_pitched{atest.d_A_, sizeof(float) * width, width,
                                       height};
  syclcompat::pitched_data d_B_pitched{atest.d_B_, sizeof(float) * width, width,
                                       height};
  sycl::id<3> pos_A(0, 0, 0);
  sycl::id<3> pos_B(0, 0, 0);
  sycl::event memcpy_ev = syclcompat::memcpy_async(
      d_A_pitched, pos_A, d_B_pitched, pos_B, {2, 2, 2});
  sycl::event host_ev = atest.launch_host_task({memcpy_ev});

  atest.check_events(memcpy_ev, host_ev);
}

void test_memset_async1() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  sycl::event memset_ev =
      syclcompat::memset_async(atest.d_C_, 1, sizeof(int) * atest.size_);
  sycl::event host_ev = atest.launch_host_task({memset_ev});

  atest.check_events(memset_ev, host_ev);
}

void test_memset_async2() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  sycl::event memset_ev =
      syclcompat::memset_async(atest.d_C_, 32, 1, sizeof(int) * 32, 4);
  sycl::event host_ev = atest.launch_host_task({memset_ev});

  atest.check_events(memset_ev, host_ev);
}

void test_memset_async3() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  assert(width * height * depth <= atest.size_);

  syclcompat::pitched_data d_A_pitched{atest.d_A_, sizeof(int) * width, width,
                                       height};
  sycl::event memset_ev =
      syclcompat::memset_async(d_A_pitched, 1, {sizeof(int) * 2, 2, 2});
  sycl::event host_ev = atest.launch_host_task({memset_ev});

  atest.check_events(memset_ev, host_ev);
}

void test_fill_event() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  sycl::event fill_ev = syclcompat::fill_async(atest.d_A_, 1.0f, atest.size_);
  sycl::event host_ev = atest.launch_host_task({fill_ev});

  atest.check_events(fill_ev, host_ev);
}

void test_combine_events() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  AsyncTest atest;

  std::vector<sycl::event> evs;
  for (int i = 0; i < 5; i++)
    evs.push_back(atest.launch_kernel());

  sycl::event combined = syclcompat::detail::combine_events(evs, atest.q_);

  using namespace sycl::info;

  // Lambda returns true if all events 'complete'
  auto all_done = [&](std::vector<sycl::event> evs) {
    return std::all_of(evs.begin(), evs.end(), [](sycl::event ev) {
      return ev.get_info<event::command_execution_status>() ==
             event_command_status::complete;
    });
  };

  event_command_status combined_status =
      combined.get_info<event::command_execution_status>();
  bool prerequisites_done = all_done(evs);

  // Check combined event remains 'submitted' if not all prerequisites completed
  if (!prerequisites_done)
    assert(combined_status == event_command_status::submitted);

  // Check all prerequisites completed once combined is completed
  while (combined_status != event_command_status::running &&
         combined_status != event_command_status::complete) {
    combined_status = combined.get_info<event::command_execution_status>();
  }
  assert(all_done(evs));
}

int main() {
  test_free_async();

  test_memcpy_async1();
  test_memcpy_async2();
  test_memcpy_async3();

  test_memset_async1();
  test_memset_async2();
  test_memset_async3();

  test_fill_event();
  test_combine_events();

  return 0;
}
