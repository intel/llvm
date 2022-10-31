//==------------ PlaceholderShortcuts.cpp --- queue unit tests -------------==//
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
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
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

class MockHandler : public handler {
public:
  MockHandler(std::shared_ptr<sycl::detail::queue_impl> Queue)
      : sycl::handler(Queue, /* IsHost */ false) {}

  int get_access_storage_size() const { return MAccStorage.size(); }
};

TEST(PlaceholderShortcuts, ShortcutsCallCorrectHandlerOverloads) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  const device Dev = Plt.get_devices()[0];

  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferWrite>(
      redefinedEnqueueMemBufferWrite);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferRead>(
      redefinedEnqueueMemBufferRead);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferCopy>(
      redefinedEnqueueMemBufferCopy);

  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);

  context Ctx{Dev};

  // Queue.copy(accessor src, shared_ptr dest);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    int Data[1];
    buffer<int, 1> Buf(Data, range<1>(1));

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(Buf);
    ASSERT_TRUE(Src.is_placeholder());

    std::shared_ptr<int> Dest = std::make_shared<int>(0);

    Queue.copy(Src, Dest);
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferReadCalled);
    // TODO: Verify accessor Src actually added to handler
  }
  // Queue.copy(shared_ptr src, accessor dest);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    int Data[1];
    buffer<int, 1> Buf(Data, range<1>(1));

    std::shared_ptr<int> Src = std::make_shared<int>(42);

    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(Buf);
    ASSERT_TRUE(Dest.is_placeholder());

    Queue.copy(Src, Dest);
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferWriteCalled);
    // TODO: Verify accessor Dest actually added to handler
  }
  // Queue.copy(accessor src, ptr* dest);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    int Data[1];
    buffer<int, 1> Buf(Data, range<1>(1));

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(Buf);
    ASSERT_TRUE(Src.is_placeholder());

    std::unique_ptr<int> Dest = std::make_unique<int>(0);

    Queue.copy(Src, Dest.get());
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferReadCalled);
    // TODO: Verify accessor Src actually added to handler
  }
  // Queue.copy(ptr* src, accessor dest);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    int Data[1];
    buffer<int, 1> Buf(Data, range<1>(1));

    std::unique_ptr<int> Src = std::make_unique<int>(42);

    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(Buf);
    ASSERT_TRUE(Dest.is_placeholder());

    Queue.copy(Src.get(), Dest);
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferWriteCalled);
    // TODO: Verify accessor Dest actually added to handler
  }
  // Queue.copy(accessor src, accessor dest);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    int SrcData[1];
    int DestData[1];
    buffer<int, 1> SrcBuf(SrcData, range<1>(1));
    buffer<int, 1> DestBuf(DestData, range<1>(1));

    accessor<int, 1, access::mode::read, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Src(SrcBuf);
    accessor<int, 1, access::mode::write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Dest(DestBuf);

    ASSERT_TRUE(Src.is_placeholder());
    ASSERT_TRUE(Dest.is_placeholder());

    Queue.copy(Src, Dest);
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferCopyCalled);
    // TODO: Verify accessor Src actually added to handler
    // TODO: Verify accessor Dest actually added to handler
  }
  // Queue.update_host(accessor acc);
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};
    MockHandler TestHandler(detail::getSyclObjImpl(Queue));

    buffer<int, 1> Buf{range<1>(1)};
    accessor<int, 1, access::mode::read_write, access::target::device,
             access::placeholder::true_t, ext::oneapi::accessor_property_list<>>
        Acc(Buf);

    // const std::type_info &ti1 = typeid(Acc);
    // std::cout << demangle(ti1.name()) << std::endl;
    ASSERT_TRUE(Acc.is_placeholder());

    int AccessorsInStorage = TestHandler.get_access_storage_size();
    Queue.update_host(Acc);
    Queue.wait();

    // TODO: Verify accessor actually added to handler
  }
  // Queue.fill<T>(accessor Dest, T src)
  {
    TestContext.reset(new TestCtx(Ctx));
    queue Queue{Dev};

    buffer<int, 1> Buf{range<1>(1)};
    accessor Acc(Buf);
    ASSERT_TRUE(Acc.is_placeholder());

    Queue.fill(Acc, 42);
    Queue.wait();

    EXPECT_TRUE(TestContext->BufferFillCalled);
    // TODO: Verify accessor Acc actually added to handler
  }
}
