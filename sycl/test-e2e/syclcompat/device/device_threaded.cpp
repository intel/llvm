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
 *  device_threaded.cpp
 *
 *  Description:
 *    Device info and selection tests
 **************************************************************************/

// The original source was under the license below:
//===-- Device.cpp -  -*- C++ -* ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux

// RUN: %{build} -lpthread -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/device.hpp>

#include "device_fixt.hpp"

// Check a thread is able to select a non-default device
void test_device_select() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  DeviceTestsFixt dtf;
  if (dtf.get_n_devices() > 1) {
    constexpr unsigned int TARGET_DEV = 1;
    unsigned int thread_dev_id{};
    std::thread other_thread{[&]() {
      syclcompat::select_device(TARGET_DEV);
      thread_dev_id = syclcompat::get_current_device_id();
    }};
    other_thread.join();
    assert(thread_dev_id == TARGET_DEV);
  } else {
    std::cout << "  Skipping, only doable with multiple devices" << std::endl;
  }
}

// Check multiple threads get same device by default
void test_threads() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  unsigned int thread_dev_id{};
  std::thread other_thread{
      [&]() { thread_dev_id = syclcompat::get_current_device_id(); }};
  other_thread.join();
  assert(thread_dev_id == syclcompat::get_current_device_id());
}

int main() {
  test_device_select();
  test_threads();

  return 0;
}
