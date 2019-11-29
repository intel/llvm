//==---- SchedulerThreadSafety.cpp --- Thread Safety unit tests ------------==//
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

template <typename T, int Dim> class TestDeadLock : public ParallelTask {
public:
  TestDeadLock(T *Data, std::size_t Size)
      : MBuffer(Data, cl::sycl::range<Dim>(Size)), MBufferSize(Size) {}

  void taskBody(size_t ThreadId) {
    auto acc = MBuffer.template get_access<sycl_read_write>();
    for (std::size_t i = 0; i < MBufferSize; ++i) {
      acc[i] = ThreadId;
      if (i == 0) {
        MMutex.lock();
        MThreadOrder.push_back(ThreadId);
        MMutex.unlock();
      }
    }
  }

  std::size_t getLastWorkingThread() { return MThreadOrder.back(); }

private:
  std::vector<std::size_t> MThreadOrder;
  cl::sycl::buffer<T, Dim> MBuffer;
  std::size_t MBufferSize;
  std::mutex MMutex;
};

class HostAccessorDeadLockTest : public ::testing::Test {};

TEST_F(HostAccessorDeadLockTest, CheckThreadOrder) {
  constexpr size_t size = 1024;
  constexpr size_t threadCount = 4;
  std::size_t data[size];
  TestDeadLock<std::size_t, 1> Task(data, size);
  Task.execute(threadCount);
  EXPECT_EQ(data[size - 1], Task.getLastWorkingThread());
}
} // namespace
