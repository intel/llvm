//==----------------- stream_impl.cpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

void stream_impl::flush() {
  // Access the stream buffer on the host. This access guarantees that kernel is
  // executed and buffer contains streamed data.
  auto HostAcc = detail::Scheduler::getInstance()
                     .StreamBuffersPool.find(this)
                     ->second.Buf.get_access<cl::sycl::access::mode::read>(
                         range<1>(BufferSize_), id<1>(OffsetSize));

  printf("%s", HostAcc.get_pointer());
  fflush(stdout);
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

