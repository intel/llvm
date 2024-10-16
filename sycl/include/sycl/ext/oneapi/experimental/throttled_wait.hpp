//===------- throttled_wait.hpp -  sleeping implementation of wait   ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <thread>

// The throttled_wait extension requires the inclusion of this header.
// If we instead want it to be included with sycl.hpp, then this defnition
// will need to be removed from here and
// added to llvm/sycl/source/feature_test.hpp.in instead.
#define SYCL_EXT_ONEAPI_THROTTLED_WAIT 1

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename Rep, typename Period>
void ext_oneapi_throttled_wait(
    sycl::event &e, const std::chrono::duration<Rep, Period> &sleep) {
  while (e.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    std::this_thread::sleep_for(sleep);
  }
  e.wait();
}

template <typename Rep, typename Period>
void ext_oneapi_throttled_wait(
    std::vector<sycl::event> &eventList,
    const std::chrono::duration<Rep, Period> &sleep) {
  for (sycl::event &e : eventList) {
    while (e.get_info<sycl::info::event::command_execution_status>() !=
           sycl::info::event_command_status::complete) {
      std::this_thread::sleep_for(sleep);
    }
    e.wait();
  }
}

template <typename Rep, typename Period>
void ext_oneapi_throttled_wait_and_throw(
    sycl::event &e, const std::chrono::duration<Rep, Period> &sleep) {
  while (e.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    std::this_thread::sleep_for(sleep);
  }
  e.wait_and_throw();
}

template <typename Rep, typename Period>
void ext_oneapi_throttled_wait_and_throw(
    std::vector<sycl::event> &eventList,
    const std::chrono::duration<Rep, Period> &sleep) {
  for (sycl::event &e : eventList) {
    while (e.get_info<sycl::info::event::command_execution_status>() !=
           sycl::info::event_command_status::complete) {
      std::this_thread::sleep_for(sleep);
    }
    e.wait_and_throw();
  }
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl