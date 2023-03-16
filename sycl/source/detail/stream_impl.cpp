//==----------------- stream_impl.cpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/buffer_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>
#include <sycl/queue.hpp>

#include <cstdio>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

stream_impl::stream_impl(size_t BufferSize, size_t MaxStatementSize,
                         handler &CGH)
    : stream_impl(BufferSize, MaxStatementSize, {}) {
  (void)CGH;
}

stream_impl::stream_impl(size_t BufferSize, size_t MaxStatementSize,
                         const property_list &PropList)
    : BufferSize_(BufferSize), MaxStatementSize_(MaxStatementSize),
      PropList_(PropList), Buf_(range<1>(BufferSize + OffsetSize + 1)),
      FlushBuf_(range<1>(MaxStatementSize + FLUSH_BUF_OFFSET_SIZE)) {
  // Additional place is allocated in the stream buffer for the offset variable
  // and the end of line symbol. Buffers are created without host pointers so
  // that they are released in a deferred manner. Disable copy back on buffer
  // destruction. Copy is scheduled as a host task which fires up as soon as
  // kernel has completed execution.
  Buf_.set_write_back(false);
  FlushBuf_.set_write_back(false);
  // Initialize stream buffer with zeros, this is needed for two reasons:
  // 1. We don't need to care about end of line when printing out
  // streamed data.
  // 2. Offset is properly initialized.
  host_accessor Acc{Buf_};
  char *Ptr = Acc.get_pointer();
  std::memset(Ptr, 0, Buf_.size());
}

// Method to provide an access to the global stream buffer
GlobalBufAccessorT stream_impl::accessGlobalBuf(handler &CGH) {
  return Buf_.get_access<sycl::access::mode::read_write>(
      CGH, range<1>(BufferSize_), id<1>(OffsetSize));
}

// Method to provide an accessor to the global flush buffer
GlobalBufAccessorT stream_impl::accessGlobalFlushBuf(handler &CGH) {
  return FlushBuf_.get_access<sycl::access::mode::read_write>(
      CGH, range<1>(MaxStatementSize_ + FLUSH_BUF_OFFSET_SIZE), id<1>(0));
}

// Method to provide an atomic access to the offset in the global stream
// buffer and offset in the flush buffer
GlobalOffsetAccessorT stream_impl::accessGlobalOffset(handler &CGH) {
  auto OffsetSubBuf = buffer<char, 1>(Buf_, id<1>(0), range<1>(OffsetSize));
  auto ReinterpretedBuf = OffsetSubBuf.reinterpret<unsigned, 1>(range<1>(2));
  return ReinterpretedBuf.get_access<sycl::access::mode::atomic>(
      CGH, range<1>(2), id<1>(0));
}

size_t stream_impl::size() const noexcept { return BufferSize_; }

size_t stream_impl::get_work_item_buffer_size() const {
  return MaxStatementSize_;
}

size_t stream_impl::get_size() const { return BufferSize_; }

size_t stream_impl::get_max_statement_size() const { return MaxStatementSize_; }

void stream_impl::initStreamHost(QueueImplPtr Queue) {
  // Real size of full flush buffer is saved only in buffer_impl field of
  // FlushBuf object.
  size_t FlushBufSize = getSyclObjImpl(FlushBuf_)->size();

  auto Q = createSyclObjFromImpl<queue>(Queue);
  Q.submit([&](handler &cgh) {
    auto FlushBufAcc = FlushBuf_.get_access<access::mode::discard_write,
                                            access::target::host_buffer>(
        cgh, range<1>(1), id<1>(0));
    cgh.host_task([=] {
      char *FlushBufPtr = FlushBufAcc.get_pointer();
      std::memset(FlushBufPtr, 0, FlushBufSize);
    });
  });
}

void stream_impl::flush(const EventImplPtr &LeadEvent) {
  // We don't want stream flushing to be blocking operation that is why submit a
  // host task to print stream buffer. It will fire up as soon as the kernel
  // finishes execution.
  auto Q = detail::createSyclObjFromImpl<queue>(
      sycl::detail::Scheduler::getInstance().getDefaultHostQueue());
  event Event = Q.submit([&](handler &cgh) {
    auto BufHostAcc =
        Buf_.get_access<access::mode::read_write, access::target::host_buffer>(
            cgh, range<1>(BufferSize_), id<1>(OffsetSize));
    // Create accessor to the flush buffer even if not using it yet. Otherwise
    // kernel will be a leaf for the flush buffer and scheduler will not be able
    // to cleanup the kernel. TODO: get rid of finalize method by using host
    // accessor to the flush buffer.
    auto FlushBufHostAcc =
        FlushBuf_
            .get_access<access::mode::read_write, access::target::host_buffer>(
                cgh);
    cgh.host_task([=] {
      printf("%s", &(BufHostAcc[0]));
      fflush(stdout);
    });
  });
  if (LeadEvent) {
    LeadEvent->attachEventToComplete(detail::getSyclObjImpl(Event));
    LeadEvent->getSubmittedQueue()->registerStreamServiceEvent(
        detail::getSyclObjImpl(Event));
  }
}

void stream_impl::flush() { flush(nullptr); }
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
