//==- BufferDestructionCheck.cpp --- check delayed destruction of buffer --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <detail/buffer_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <gmock/gmock.h>

class FairMockScheduler : public sycl::detail::Scheduler {
public:
  // FairMockScheduler() : Scheduler()
  // {
  //     // ON_CALL(*this, deferMemObjRelease(_)).
  //     //     WillByDefault(Invoke([&](qcsinternal::Duration timeout) {
  //     //     return qcsinternal::PidNamedEvent::TryLockFor(timeout);
  //     // }));
  // }
  MOCK_METHOD1(deferMemObjRelease,
               void(const std::shared_ptr<sycl::detail::SYCLMemObjI> &));
};

class BufferDestructionCheck : public ::testing::Test {
public:
  BufferDestructionCheck() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    if (Plt.is_host()) {
      std::cout << "Not run due to host-only environment\n";
      GTEST_SKIP();
    }
    MockSchedulerPtr = new FairMockScheduler();
    sycl::detail::GlobalHandler::instance().attachScheduler(
        dynamic_cast<sycl::detail::Scheduler *>(MockSchedulerPtr));
    // Mock.redefine<sycl::detail::PiApiKind::piMemBufferCreate>(
    //     redefinedMemBufferCreate);
    // Mock.redefine<sycl::detail::PiApiKind::piDeviceGetInfo>(
    //     redefinedDeviceGetInfo);
  }
  void TearDown() override {
    sycl::detail::GlobalHandler::instance().attachScheduler(NULL);
  }

  inline void
  CheckBufferDestruction(std::shared_ptr<sycl::detail::buffer_impl> BufImpl,
                         bool ShouldBeDeferred) {
    ASSERT_NE(BufImpl, nullptr);
    const std::function<bool(
        const std::shared_ptr<sycl::detail::SYCLMemObjI> &)>
        checkerNotEqual =
            [&BufImpl](
                const std::shared_ptr<sycl::detail::SYCLMemObjI> &memObj) {
              return BufImpl.get() != memObj.get();
            };
    const std::function<bool(
        const std::shared_ptr<sycl::detail::SYCLMemObjI> &)>
        checkerEqual =
            [&BufImpl](
                const std::shared_ptr<sycl::detail::SYCLMemObjI> &memObj) {
              return BufImpl.get() == memObj.get();
            };
    if (ShouldBeDeferred) {
      testing::Sequence S;
      // first is check that explicitly created buffer is deferred
      EXPECT_CALL(*MockSchedulerPtr,
                  deferMemObjRelease(testing::Truly(checkerEqual)))
          .Times(1)
          .InSequence(S)
          .RetiresOnSaturation();
      // we have two queues - non host and host queue. Currently queue contains
      // its own buffer as class member, buffer as used for assert handling.
      // those buffers also created with size only so it also to be deferred on
      // deletion.
      EXPECT_CALL(*MockSchedulerPtr, deferMemObjRelease(testing::_))
          .Times(/*testing::AnyNumber()*/ 2)
          .InSequence(S);
    } else {
      // buffer created above should not be deferred on deletion because has non
      // default allocator
      EXPECT_CALL(*MockSchedulerPtr,
                  deferMemObjRelease(testing::Truly(checkerNotEqual)))
          .Times(testing::AnyNumber());
    }
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
  FairMockScheduler *MockSchedulerPtr;
};

// Test that buffer_location was passed correctly
TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefault) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    sycl::buffer<int, 1> Buf(1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSizeOnlyDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    using AllocatorTypeTest = sycl::buffer_allocator<int>;
    AllocatorTypeTest allocator;
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithRawHostPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    int InitialVal = 8;
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(&InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithConstRawHostPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    const int InitialVal = 8;
    sycl::buffer<int, 1> Buf(&InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck,
       BufferWithConstRawHostPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    const int InitialVal = 8;
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(&InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithContainer) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithContainerWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(data, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtr) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int> InitialVal(new int(5));
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtrWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int> InitialVal(new int(5));
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(InitialVal, 1, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithSharedPtrArray) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int[]> InitialVal(new int[2]);
    sycl::buffer<int, 1> Buf(InitialVal, 1);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck,
       BufferWithSharedPtrArrayWithNonDefaultAllocator) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::shared_ptr<int[]> InitialVal(new int[2]);
    using AllocatorTypeTest =
        sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    AllocatorTypeTest allocator(Q);
    sycl::buffer<int, 1, AllocatorTypeTest> Buf(InitialVal, 2, allocator);
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, false);
  }
}

TEST_F(BufferDestructionCheck, BufferWithIterators) {
  sycl::context Context{Plt};
  sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
  {
    std::vector<int> data{3, 4};
    sycl::buffer<int, 1> Buf(data.begin(), data.end());
    std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
        sycl::detail::getSyclObjImpl(Buf);
    CheckBufferDestruction(BufImpl, true);
  }
}

// TEST_F(BufferDestructionCheck, BufferWithIteratorsWithNonDefaultAllocator) {
//   sycl::context Context{Plt};
//   sycl::queue Q = sycl::queue{Context, sycl::default_selector{}};
//   {
//     std::vector<int> data{3, 4};
//     using AllocatorTypeTest = sycl::usm_allocator<int,
//     sycl::usm::alloc::shared>; AllocatorTypeTest allocator(Q);
//     sycl::buffer<int, 1, AllocatorTypeTest> Buf(data.begin(), data.end(),
//     allocator); std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
//         sycl::detail::getSyclObjImpl(Buf);
//     CheckBufferDestruction(BufImpl, false);
//   }
// }