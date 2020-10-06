//==----------------- stream_impl.cpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/queue.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/stream_impl.hpp>

#include <cstdio>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

stream_impl::stream_impl(size_t BufferSize, size_t MaxStatementSize,
                         handler &CGH)
    : BufferSize_(BufferSize), MaxStatementSize_(MaxStatementSize) {
  // We need to store stream buffers in the scheduler because they need to be
  // alive after submitting the kernel. They cannot be stored in the stream
  // object because it causes loop dependency between objects and results in
  // memory leak.
  // Allocate additional place in the stream buffer for the offset variable and
  // the end of line symbol.
  detail::Scheduler::getInstance().allocateStreamBuffers(
      this, BufferSize + OffsetSize + 1 /* size of the stream buffer */,
      MaxStatementSize /* size of the flush buffer */);
}

// Method to provide an access to the global stream buffer
GlobalBufAccessorT stream_impl::accessGlobalBuf(handler &CGH) {
  return detail::Scheduler::getInstance()
      .StreamBuffersPool.find(this)
      ->second.Buf.get_access<cl::sycl::access::mode::read_write>(
          CGH, range<1>(BufferSize_), id<1>(OffsetSize));
}

// Method to provide an accessor to the global flush buffer
GlobalBufAccessorT stream_impl::accessGlobalFlushBuf(handler &CGH) {
  return detail::Scheduler::getInstance()
      .StreamBuffersPool.find(this)
      ->second.FlushBuf.get_access<cl::sycl::access::mode::read_write>(
          CGH, range<1>(MaxStatementSize_), id<1>(0));
}

// Method to provide an atomic access to the offset in the global stream
// buffer and offset in the flush buffer
GlobalOffsetAccessorT stream_impl::accessGlobalOffset(handler &CGH) {
  auto OffsetSubBuf = buffer<char, 1>(
      detail::Scheduler::getInstance().StreamBuffersPool.find(this)->second.Buf,
      id<1>(0), range<1>(OffsetSize));
  auto ReinterpretedBuf = OffsetSubBuf.reinterpret<unsigned, 1>(range<1>(2));
  return ReinterpretedBuf.get_access<cl::sycl::access::mode::atomic>(
      CGH, range<1>(2), id<1>(0));
}
size_t stream_impl::get_size() const { return BufferSize_; }

size_t stream_impl::get_max_statement_size() const { return MaxStatementSize_; }

std::once_flag flag;

void stream_impl::flush() {
  // We don't want stream flushing to be blocking operation that is why submit a
  // host task to print stream buffer.
  queue &Q = cl::sycl::detail::Scheduler::getInstance().getDefaultHostQueue();
  Q.submit([&](handler &cgh) {
    auto HostAcc =
        detail::Scheduler::getInstance()
            .StreamBuffersPool.find(this)
            ->second.Buf
            .get_access<access::mode::read, access::target::host_buffer>(
                cgh, range<1>(BufferSize_), id<1>(OffsetSize));
    cgh.codeplay_host_task([=]() {
      printf("%s", HostAcc.get_pointer());
      fflush(stdout);
    });
  });

  // Register a library cleanup function to deallocate stream buffers, we need
  // to do it only once. Deallocation of the buffer will also guarantee that
  // copy back and printing is completed.
  std::call_once(flag, []() {
    std::atexit(
        []() { detail::Scheduler::getInstance().deallocateStreamBuffers(); });
  });
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

