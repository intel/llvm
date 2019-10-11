//==------------------- stream.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/stream.hpp>

namespace cl {
namespace sycl {

stream::stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH)
    : impl(std::make_shared<detail::stream_impl>(BufferSize, MaxStatementSize,
                                                 CGH)),
      GlobalBuf(impl->accessGlobalBuf(CGH)),
      GlobalOffset(impl->accessGlobalOffset(CGH)),
      // Allocate pool of flush buffers, which contains space for each work item
      // in the work group
      FlushBufs(MaxStatementSize, CGH),
      // Offset of the WI's flush buffer in the pool, we need atomic access to
      // this offset to differentiate work items so that output from work items
      // is not mixed
      WIOffsetAcc(range<1>(1), CGH),
      FlushSize(impl->accessFlushBufferSize(CGH)),
      FlushBufferSize(MaxStatementSize) {

  // Save stream implementation in the handler so that stream will be alive
  // during kernel execution
  CGH.addStream(impl);

  // Set flag identifying that created local accessor has perWI size. Accessor
  // will be resized in SYCL RT when number of work items per work group will be
  // available. Local memory size and max work group size is provided to the
  // accessor. This info is used to do allocation if work group size is not
  // provided by user.
  detail::getSyclObjImpl(FlushBufs)->PerWI = true;
  detail::getSyclObjImpl(FlushBufs)->LocalMemSize =
      CGH.MQueue->get_device().get_info<info::device::local_mem_size>();
  detail::getSyclObjImpl(FlushBufs)->MaxWGSize =
      CGH.MQueue->get_device().get_info<info::device::max_work_group_size>();
}

size_t stream::get_size() const { return impl->get_size(); }

size_t stream::get_max_statement_size() const {
  return impl->get_max_statement_size();
}

bool stream::operator==(const stream &RHS) const { return (impl == RHS.impl); }

bool stream::operator!=(const stream &RHS) const { return !(impl == RHS.impl); }

} // namespace sycl
} // namespace cl

