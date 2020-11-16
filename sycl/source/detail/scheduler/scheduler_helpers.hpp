//==---------- scheduler_helpers.hpp - SYCL standard header file -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/queue.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void initStream(StreamImplPtr Stream, QueueImplPtr Queue) {
  auto StreamBuf =
      Scheduler::getInstance().StreamBuffersPool.find(Stream.get());
  assert((StreamBuf != Scheduler::getInstance().StreamBuffersPool.end()) &&
         "Stream is unexpectedly not found in pool.");

  auto &FlushBuf = StreamBuf->second->FlushBuf;
  // Only size of buffer_impl object has been resized.
  // Value of Range field of FlushBuf instance is still equal to
  // MaxStatementSize only.
  size_t FlushBufSize = getSyclObjImpl(FlushBuf)->get_count();

  auto Q = createSyclObjFromImpl<queue>(Queue);
  Q.submit([&](handler &cgh) {
    auto FlushBufAcc = FlushBuf.get_access<access::mode::discard_write,
                                           access::target::host_buffer>(
        cgh, range<1>(FlushBufSize), id<1>(0));
    cgh.codeplay_host_task([=] {
      char *FlushBufPtr = FlushBufAcc.get_pointer();
      std::memset(FlushBufPtr, 0, FlushBufAcc.get_size());
    });
  });
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
