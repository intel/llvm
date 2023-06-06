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
 *  MemoryAsync.cpp
 *
 *  Description:
 *    Asynchronous memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ MemoryAsync.cpp------------------- -*- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// Tests for the sycl::events returned from compat::*Async API calls

#include <gtest/gtest.h>
#include <stdio.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>
#define WG_SIZE 256
#define NUM_WG 32

using namespace sycl::ext::oneapi::experimental;

void VectorAddKernel(float *A, float *B, float *C) {
  auto id = compat::local_id::x();
  A[id] = id + 1.0f;
  B[id] = id + 1.0f;
  C[id] = A[id] + B[id];
}

// Fixture to set up & launch a kernel to depend on, or
// a host_task which depends on something else
class AsyncTest : public ::testing::Test {
protected:
  AsyncTest()
      : q_{compat::get_default_queue()}, grid_{NUM_WG}, thread_{WG_SIZE},
        size_{WG_SIZE * NUM_WG} {
    d_A = sycl::malloc_device<float>(size_, q_);
    d_B = sycl::malloc_device<float>(size_, q_);
    d_C = sycl::malloc_device<float>(size_, q_);
  }

  ~AsyncTest() {
    sycl::free(d_A, q_);
    sycl::free(d_B, q_);
    sycl::free(d_C, q_);
  }
  sycl::event launch_kernel() {
    return launch<VectorAddKernel>(grid_, thread_, q_, d_A, d_B, d_C);
  }

  sycl::event launch_host_task(std::vector<sycl::event> dep_events) {
    return q_.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.host_task([]() {});
    });
  }

  // Check that a dependent event (e2) doesn't start until after the dependee
  // (e1)
  void check_events(sycl::event e1, sycl::event e2) {
    using namespace sycl::info;

    event_command_status e2_status =
        e2.get_info<event::command_execution_status>();
    event_command_status e1_status =
        e1.get_info<event::command_execution_status>();

    // Check event 2 hasn't started iff event 1 hasn't finished
    if (e1_status != event_command_status::complete) {
      EXPECT_EQ(e2_status, event_command_status::submitted);
    }

    // Once event 2 is running, check event 1 has finished
    while (e2_status != event_command_status::running &&
           e2_status != event_command_status::complete) {
      e2_status = e2.get_info<event::command_execution_status>();
    }
    EXPECT_EQ(e1.get_info<event::command_execution_status>(),
              event_command_status::complete);
  }

  sycl::queue q_;
  compat::dim3 const grid_;
  compat::dim3 const thread_;
  float *d_A;
  float *d_B;
  float *d_C;
  size_t size_;
};

// free_async is a host task, so we are really testing the event dependency here
TEST_F(AsyncTest, free_async) {
  float *d_D = (float *)compat::malloc(sizeof(float));
  sycl::event kernel_ev = launch_kernel();
  sycl::event free_ev = compat::free_async({d_D}, {kernel_ev});

  check_events(kernel_ev, free_ev);
}

// The following tests are simply testing (as best possible) that
// the sycl::event returned from *Async really corresponds to the task
// We don't check that the memory operation does what it's supposed to,
// this is tested elsewhere.
TEST_F(AsyncTest, memcpy_async1) {
  sycl::event memcpy_ev = compat::memcpy_async(d_A, d_C, sizeof(float) * size_);
  sycl::event host_ev = launch_host_task({memcpy_ev});

  check_events(memcpy_ev, host_ev);
}

TEST_F(AsyncTest, memcpy_async2) {
  sycl::event memcpy_ev = compat::memcpy_async(d_A, 32, d_C, 32, 32, 4);
  sycl::event host_ev = launch_host_task({memcpy_ev});

  check_events(memcpy_ev, host_ev);
}

TEST_F(AsyncTest, memcpy_async3) {
  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  assert(width * height * depth <= size_);

  compat::pitched_data d_A_pitched{d_A, sizeof(float) * width, width, height};
  compat::pitched_data d_B_pitched{d_B, sizeof(float) * width, width, height};
  sycl::id<3> pos_A(0, 0, 0);
  sycl::id<3> pos_B(0, 0, 0);
  sycl::event memcpy_ev =
      compat::memcpy_async(d_A_pitched, pos_A, d_B_pitched, pos_B, {2, 2, 2});
  sycl::event host_ev = launch_host_task({memcpy_ev});

  check_events(memcpy_ev, host_ev);
}

TEST_F(AsyncTest, memset_async1) {
  sycl::event memset_ev = compat::memset_async(d_C, 1, sizeof(int) * size_);
  sycl::event host_ev = launch_host_task({memset_ev});

  check_events(memset_ev, host_ev);
}

TEST_F(AsyncTest, memset_async2) {
  sycl::event memset_ev = compat::memset_async(d_C, 32, 1, sizeof(int) * 32, 4);
  sycl::event host_ev = launch_host_task({memset_ev});

  check_events(memset_ev, host_ev);
}

TEST_F(AsyncTest, memset_async3) {
  size_t width = 4;
  size_t height = 4;
  size_t depth = 4;
  assert(width * height * depth <= size_);

  compat::pitched_data d_A_pitched{d_A, sizeof(int) * width, width, height};
  sycl::event memset_ev =
      compat::memset_async(d_A_pitched, 1, {sizeof(int) * 2, 2, 2});
  sycl::event host_ev = launch_host_task({memset_ev});

  check_events(memset_ev, host_ev);
}

TEST_F(AsyncTest, fill_event) {
  sycl::event fill_ev = compat::fill_async(d_A, 1.0f, size_);
  sycl::event host_ev = launch_host_task({fill_ev});

  check_events(fill_ev, host_ev);
}

TEST_F(AsyncTest, CombineEvents) {
  std::vector<sycl::event> evs;
  for (int i = 0; i < 5; i++)
    evs.push_back(launch_kernel());

  sycl::event combined = compat::detail::combine_events(evs, q_);

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
    EXPECT_EQ(combined_status, event_command_status::submitted);

  // Check all prerequisites completed once combined is completed
  while (combined_status != event_command_status::running &&
         combined_status != event_command_status::complete) {
    combined_status = combined.get_info<event::command_execution_status>();
  }
  EXPECT_TRUE(all_done(evs));
}
