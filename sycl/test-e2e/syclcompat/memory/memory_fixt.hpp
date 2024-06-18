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
 *  memory_fixt.hpp
 *
 *  Description:
 *    Memory content fixtures for the Memory functionality tests
 **************************************************************************/

#pragma once

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

constexpr size_t WG_SIZE = 256;
constexpr size_t NUM_WG = 32;

// Fixture to set up & launch a kernel to depend on, or
// a host_task which depends on something else
class AsyncTest {
public:
  AsyncTest()
      : q_{syclcompat::get_default_queue()}, grid_{NUM_WG}, thread_{WG_SIZE},
        size_{WG_SIZE * NUM_WG} {
    d_A_ = sycl::malloc_device<float>(size_, q_);
    d_B_ = sycl::malloc_device<float>(size_, q_);
    d_C_ = sycl::malloc_device<float>(size_, q_);
  }

  ~AsyncTest() {
    sycl::free(d_A_, q_);
    sycl::free(d_B_, q_);
    sycl::free(d_C_, q_);
  }
  sycl::event launch_kernel() {
    auto &dd_A = d_A_;
    auto &dd_B = d_B_;
    auto &dd_C = d_C_;
    return q_.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(size_, [=](sycl::id<1> id) {
        dd_A[id] = static_cast<float>(id) + 1.0f;
        dd_B[id] = static_cast<float>(id) + 1.0f;
        dd_C[id] = dd_A[id] + dd_B[id];
      });
    });
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
      assert(e2_status == event_command_status::submitted);
    }

    // Once event 2 is finished, check event 1 has finished
    while (e2_status != event_command_status::complete) {
      e2_status = e2.get_info<event::command_execution_status>();
    }
    assert(e1.get_info<event::command_execution_status>() ==
           event_command_status::complete);
  }

  sycl::queue q_;
  syclcompat::dim3 const grid_;
  syclcompat::dim3 const thread_;
  float *d_A_;
  float *d_B_;
  float *d_C_;
  size_t size_;
};

template <typename T> bool should_skip(const sycl::device &dev) {
  bool skip = false;
  if (!dev.has(sycl::aspect::fp64) && std::is_same_v<T, double>) {
    std::cout << "  sycl::aspect::fp64 not supported by the SYCL device."
              << std::endl;
    skip = true;
  }
  if (!dev.has(sycl::aspect::fp16) && std::is_same_v<T, sycl::half>) {
    std::cout << "  sycl::aspect::fp16 not supported by the SYCL device."
              << std::endl;
    skip = true;
  }
  return skip;
}

// USM Tests Helpers
// Fixture to set up & launch testing kernel
template <typename T> struct USMTest {
  USMTest()
      : q_{syclcompat::get_default_queue()}, grid_{NUM_WG}, thread_{WG_SIZE},
        size_{WG_SIZE * NUM_WG},
        skip{should_skip<T>(syclcompat::get_current_device())} {}

  void launch_kernel() {
    auto &dd_A = data;
    return q_
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(
              size_, [=](sycl::id<1> id) { dd_A[id] = static_cast<int>(id); });
        })
        .wait();
  }

  // Check result is identity vector
  // Handles memcpy for USM device alloc
  void check_result() {
    sycl::usm::alloc ptr_type = sycl::get_pointer_type(data, q_.get_context());
    assert(ptr_type != sycl::usm::alloc::unknown);

    T *result;
    if (ptr_type == sycl::usm::alloc::device) {
      result = static_cast<T *>(std::malloc(sizeof(T) * size_));
      syclcompat::memcpy(result, data, sizeof(T) * size_);
    } else {
      result = data;
    }

    for (size_t i = 0; i < size_; i++) {
      assert(result[i] == static_cast<T>(i));
    }

    if (ptr_type == sycl::usm::alloc::device)
      std::free(result);
  }

  sycl::queue q_;
  syclcompat::dim3 const grid_;
  syclcompat::dim3 const thread_;
  T *data;
  size_t size_;
  bool skip;
};

template <auto F> class LocalMemTest {
public:
  LocalMemTest(syclcompat::dim3 grid, syclcompat::dim3 threads)
      : grid_{grid}, threads_{threads}, size_{grid_.size() * threads_.size()},
        host_data_(size_) {
    data_ = (int *)syclcompat::malloc(size_ * sizeof(int));
  };
  ~LocalMemTest() { syclcompat::free(data_); };

  template <typename Lambda, typename... Args>
  void launch_test(Lambda checker, Args... args) {
    syclcompat::launch<F>(grid_, threads_, data_, args...);
    syclcompat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    checker(host_data_);
  }

private:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  size_t size_;
  sycl::queue q_;
  int *data_;
  std::vector<int> host_data_;
  using CheckLambda = std::function<void(std::vector<int>)>;
};
