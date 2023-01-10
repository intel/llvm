//==----- HostAccessorDeadLock.cpp --- Thread Safety unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadUtils.h"
#include <gtest/gtest.h>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace {
constexpr auto sycl_read_write = sycl::access::mode::read_write;

TEST(HostAccessorDeadLockTest, CheckThreadOrder) {
  constexpr std::size_t size = 1;
  constexpr std::size_t threadCount = 4, launchCount = 5;

  {
    sycl::buffer<std::size_t, 1> buffer(size);

    auto testLambda = [&](std::size_t threadId) {
      auto acc = buffer.get_access<sycl_read_write>();
    };

    for (std::size_t k = 0; k < launchCount; ++k) {
      ThreadPool MPool(threadCount, testLambda);
    }
  }
}
} // namespace
