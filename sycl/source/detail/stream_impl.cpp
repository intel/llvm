//==----------------- stream_impl.cpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/stream_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <cstdio>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

stream_impl::stream_impl(size_t BufferSize, size_t MaxStatementSize,
                         handler &CGH)
    : BufferSize_(BufferSize), MaxStatementSize_(MaxStatementSize),
      // Allocate additional place for the offset variable and the end of line
      // symbol. Initialize buffer with zeros, this is needed for two reasons:
      // 1. We don't need to care about end of line when printing out streamed
      // data.
      // 2. Offset is properly initialized.
      Data(BufferSize + OffsetSize + 1, 0),
      Buf(Data.data(), range<1>(BufferSize + OffsetSize + 1),
          {property::buffer::use_host_ptr()}),

      FlushBuf(range<1>(MaxStatementSize)) {}

size_t stream_impl::get_size() const { return BufferSize_; }

size_t stream_impl::get_max_statement_size() const { return MaxStatementSize_; }

void stream_impl::flush() {
  // Access the stream buffer on the host. This access guarantees that kernel is
  // executed and buffer contains streamed data.
  auto HostAcc = Buf.get_access<cl::sycl::access::mode::read>(
      range<1>(BufferSize_), id<1>(OffsetSize));

  printf("%s", HostAcc.get_pointer());
  fflush(stdout);
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

