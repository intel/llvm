//==-------------- ShortcutFunctions.cpp --- queue unit tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>
#include <sycl/sycl.hpp>

#include <memory>

using namespace sycl;

namespace {
struct TestCtx {
  bool BufferFillCalled = false;
  bool BufferReadCalled = false;
  bool BufferWriteCalled = false;
  bool BufferCopyCalled = false;
};
} // namespace

static std::unique_ptr<TestCtx> TestContext;

pi_result redefinedEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                         pi_bool blocking_write, size_t offset,
                                         size_t size, const void *ptr,
                                         pi_uint32 num_events_in_wait_list,
                                         const pi_event *event_wait_list,
                                         pi_event *event) {
  TestContext->BufferWriteCalled = true;
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemBufferRead(pi_queue queue, pi_mem buffer,
                                        pi_bool blocking_read, size_t offset,
                                        size_t size, void *ptr,
                                        pi_uint32 num_events_in_wait_list,
                                        const pi_event *event_wait_list,
                                        pi_event *event) {
  TestContext->BufferReadCalled = true;
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemBufferCopy(pi_queue command_queue,
                                        pi_mem src_buffer, pi_mem dst_buffer,
                                        size_t src_offset, size_t dst_offset,
                                        size_t size,
                                        pi_uint32 num_events_in_wait_list,
                                        const pi_event *event_wait_list,
                                        pi_event *event) {
  TestContext->BufferCopyCalled = true;
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
                                        const void *pattern,
                                        size_t pattern_size, size_t offset,
                                        size_t size,
                                        pi_uint32 num_events_in_wait_list,
                                        const pi_event *event_wait_list,
                                        pi_event *event) {
  TestContext->BufferFillCalled = true;
  return PI_SUCCESS;
}

TEST(ShortcutFunctions, ShortcutsCallCorrectPIFunctions) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferWrite>(
      redefinedEnqueueMemBufferWrite);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferRead>(
      redefinedEnqueueMemBufferRead);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferCopy>(
      redefinedEnqueueMemBufferCopy);

  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);

  context Ctx(Plt);
  queue Q{Ctx, default_selector()};

  constexpr std::size_t Size = 1;

  // Queue.copy(accessor src, shared_ptr dest);
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(Buf);
    ASSERT_TRUE(Src.is_placeholder());

    std::shared_ptr<int> Dest = std::make_shared<int>(0);

    Q.copy(Src, Dest);
    Q.wait();

    EXPECT_TRUE(TestContext->BufferReadCalled);
  }

  // Queue.copy(shared_ptr src, accessor dest);
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    std::shared_ptr<int> Src = std::make_shared<int>(42);

    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(Buf);
    ASSERT_TRUE(Dest.is_placeholder());

    Q.copy(Src, Dest);
    Q.wait();

    EXPECT_TRUE(TestContext->BufferWriteCalled);
  }

  // Queue.copy(accessor src, ptr* dest);
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(Buf);
    ASSERT_TRUE(Src.is_placeholder());

    std::unique_ptr<int> Dest = std::make_unique<int>(0);

    Q.copy(Src, Dest.get());
    Q.wait();

    EXPECT_TRUE(TestContext->BufferReadCalled);
  }

  // Queue.copy(ptr* src, accessor dest);
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    std::unique_ptr<int> Src = std::make_unique<int>(42);

    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(Buf);
    ASSERT_TRUE(Dest.is_placeholder());

    Q.copy(Src.get(), Dest);
    Q.wait();

    EXPECT_TRUE(TestContext->BufferWriteCalled);
  }

  // Queue.copy(accessor src, accessor dest);
  {
    TestContext.reset(new TestCtx());

    int SrcData[Size];
    buffer<int> SrcBuf(SrcData, Size);

    int DestData[Size];
    buffer<int> DestBuf(DestData, Size);

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(SrcBuf);
    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(DestBuf);

    ASSERT_TRUE(Src.is_placeholder());
    ASSERT_TRUE(Dest.is_placeholder());

    Q.copy(Src, Dest);
    Q.wait();

    EXPECT_TRUE(TestContext->BufferCopyCalled);
  }

  // Queue.update_host(accessor acc);
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    accessor<int, 1, access::mode::read_write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Acc(Buf);

    ASSERT_TRUE(Acc.is_placeholder());

    Q.update_host(Acc);
    Q.wait();

    // No PI functions expected.
  }

  // Queue.fill<T>(accessor Dest, T src)
  {
    TestContext.reset(new TestCtx());

    int Data[Size];
    buffer<int> Buf(Data, Size);

    accessor<int, 1, access::mode::read_write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Acc(Buf);
    ASSERT_TRUE(Acc.is_placeholder());

    Q.fill(Acc, 42);
    Q.wait();

    EXPECT_TRUE(TestContext->BufferFillCalled);
  }
}
