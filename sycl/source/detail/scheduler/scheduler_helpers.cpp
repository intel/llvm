//==-- scheduler_helpers.cpp - SYCL Scheduler helper functions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/queue.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/scheduler/scheduler_helpers.hpp>
#include <detail/stream_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void initStream(StreamImplPtr Stream, QueueImplPtr Queue) {
  Scheduler::StreamBuffers *StrBufs{};

  {
    std::lock_guard<std::recursive_mutex> lock(
        Scheduler::getInstance().StreamBuffersPoolMutex);

    auto StreamBuf =
        Scheduler::getInstance().StreamBuffersPool.find(Stream.get());
    assert((StreamBuf != Scheduler::getInstance().StreamBuffersPool.end()) &&
           "Stream is unexpectedly not found in pool.");

    StrBufs = StreamBuf->second;
  }

  assert(StrBufs && "No buffers for a stream.");

  // Real size of full flush buffer is saved only in buffer_impl field of
  // FlushBuf object.
  size_t FlushBufSize = getSyclObjImpl(StrBufs->FlushBuf)->size();

  auto Q = createSyclObjFromImpl<queue>(Queue);
  Q.submit([&](handler &cgh) {
    auto FlushBufAcc =
        StrBufs->FlushBuf.get_access<access::mode::discard_write,
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
