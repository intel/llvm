//==----- HostAccessorDeadLock.cpp --- Thread Safety unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadUtils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <mutex>
#include <vector>

namespace {
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;

TEST(HostAccessorDeadLockTest, CheckThreadOrder) {
  constexpr std::size_t size = 1024;
  constexpr std::size_t threadCount = 4;
  std::size_t data[size];
  std::size_t lastThreadNum = -1, launchCount = 5;

  {
    ThreadPool MPool;
    std::vector<std::size_t> threadOrder;
    cl::sycl::buffer<std::size_t, 1> buffer(data, size);
    std::mutex mutex;

    auto testLambda = [&](std::size_t threadId) {
      auto acc = buffer.get_access<sycl_read_write>();
      for (std::size_t i = 0; i < size; ++i) {
        acc[i] = threadId;
        if (i == 0) {
          std::lock_guard<std::mutex> lock(mutex);
          threadOrder.push_back(threadId);
        }
      }
    };

    for (std::size_t k = 0; k < launchCount; ++k) {
      MPool.clear();
      for (std::size_t i = 0; i < threadCount; ++i)
        MPool.enqueue(testLambda, i);
      MPool.wait();
    }

    lastThreadNum = threadOrder.back();
  }

  EXPECT_EQ(data[size - 1], lastThreadNum);
}
} // namespace
