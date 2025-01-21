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
 *  device_profiling.cpp
 *
 *  Description:
 *    Tests for the enable_profiling property paths
 **************************************************************************/

// RUN: %{build} -DSYCLCOMPAT_PROFILING_ENABLED=1 -o %t-profiling.out
// RUN: %{run} %t-profiling.out
// RUN: %{build} -o %t-no-profiling.out
// RUN: %{run} %t-no-profiling.out

#include <syclcompat/device.hpp>

#ifdef SYCLCOMPAT_PROFILING_ENABLED
void test_event_profiling() {
  sycl::queue q = syclcompat::get_default_queue();

  if (!q.get_device().has(sycl::aspect::queue_profiling)) {
    std::cout << "Device does not have aspect::queue_profiling, skipping."
              << std::endl;
    return;
  }

  assert(q.has_property<sycl::property::queue::enable_profiling>());

  q = sycl::queue{q.get_device(), sycl::property::queue::enable_profiling()};
  auto event = q.submit([&](sycl::handler &cgh) { cgh.single_task([=]() {}); });
  event.get_profiling_info<sycl::info::event_profiling::command_end>();
}
#else
void test_no_event_profiling() {
  sycl::queue q = syclcompat::get_default_queue();

  if (!q.get_device().has(sycl::aspect::queue_profiling)) {
    std::cout << "Device does not have aspect::queue_profiling, skipping."
              << std::endl;
    return;
  }

  assert(!q.has_property<sycl::property::queue::enable_profiling>());
}
#endif

int main() {
#ifdef SYCLCOMPAT_PROFILING_ENABLED
  test_event_profiling();
#else
  test_no_event_profiling();
#endif

  return 0;
}
