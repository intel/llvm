//==-- ReusableEventsFixture.cpp --- Mock backend for reusable events -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <helpers/MockDeviceImage.hpp>
#include <helpers/ScopedEnvVar.hpp>

namespace {

sycl::unittest::MockDeviceImage DevImage =
    sycl::unittest::generateDefaultImage({"BindingTestKernel"});
sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

// The host task thread pool is created once per process, with a single thread
// by default. Several tests block more than one host task at a time, with
// further host tasks (and dependency bridges) behind them, so the pool is
// sized before the first host task of the process reads the setting.
const bool ThreadPoolSized = [] {
  sycl::unittest::set_env("SYCL_QUEUE_THREAD_POOL_SIZE", "8");
  return true;
}();

} // anonymous namespace
