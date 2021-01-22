//==------------------- stream.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/stream.hpp>
#include <detail/queue_impl.hpp>
#include <detail/stream_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Maximum possible size of a flush buffer statement in bytes
static constexpr size_t MAX_STATEMENT_SIZE =
    (1 << (CHAR_BIT * detail::FLUSH_BUF_OFFSET_SIZE)) - 1;

stream::stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH)
    : impl(std::make_shared<detail::stream_impl>(BufferSize, MaxStatementSize,
                                                 CGH)),
      GlobalBuf(impl->accessGlobalBuf(CGH)),
      GlobalOffset(impl->accessGlobalOffset(CGH)),
      // Allocate the flush buffer, which contains space for each work item
      GlobalFlushBuf(impl->accessGlobalFlushBuf(CGH)),
      FlushBufferSize(MaxStatementSize + detail::FLUSH_BUF_OFFSET_SIZE) {
  if (MaxStatementSize > MAX_STATEMENT_SIZE) {
    throw sycl::invalid_parameter_error(
        "Maximum statement size exceeds limit of " +
            std::to_string(MAX_STATEMENT_SIZE) + " bytes.",
        PI_INVALID_VALUE);
  }

  // Save stream implementation in the handler so that stream will be alive
  // during kernel execution
  CGH.addStream(impl);

  // Set flag identifying that created accessor has perWI size. Accessor
  // will be resized in SYCL RT when number of work items will be
  // available.
  detail::getSyclObjImpl(GlobalFlushBuf)->PerWI = true;
}

size_t stream::get_size() const { return impl->get_size(); }

size_t stream::get_max_statement_size() const {
  return impl->get_max_statement_size();
}

bool stream::operator==(const stream &RHS) const { return (impl == RHS.impl); }

bool stream::operator!=(const stream &RHS) const { return !(impl == RHS.impl); }

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

