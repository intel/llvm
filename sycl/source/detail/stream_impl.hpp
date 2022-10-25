//==----------------- stream_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/handler.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/stream.hpp>

#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class __SYCL_EXPORT stream_impl {
public:
  // TODO: This constructor is unused.
  // To be removed when API/ABI changes are allowed.
  stream_impl(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  stream_impl(size_t BufferSize, size_t MaxStatementSize,
              const property_list &PropList);

  // Method to provide an access to the global stream buffer
  GlobalBufAccessorT accessGlobalBuf(handler &CGH);

  // Method to provide an accessor to the global flush buffer
  GlobalBufAccessorT accessGlobalFlushBuf(handler &CGH);

  // Method to provide an atomic access to the offset in the global stream
  // buffer and offset in the flush buffer
  GlobalOffsetAccessorT accessGlobalOffset(handler &CGH);

  // Enqueue task to copy stream buffer to the host and print the contents
  // The host task event is then registered for post processing in the
  // LeadEvent as well as in queue LeadEvent associated with.
  void flush(const EventImplPtr &LeadEvent);

  // Enqueue task to copy stream buffer to the host and print the contents
  // Remove during next ABI breaking window
  void flush();

  size_t get_size() const;

  size_t get_max_statement_size() const;

  template <typename propertyT> bool has_property() const noexcept {
    return PropList_.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return PropList_.get_property<propertyT>();
  }

private:
  // Size of the stream buffer
  size_t BufferSize_;

  // Maximum number of symbols which could be streamed from the beginning of a
  // statement till the semicolon
  unsigned MaxStatementSize_;

  // Property list
  property_list PropList_;

  // Additinonal memory is allocated in the beginning of the stream buffer for
  // 2 variables: offset in the stream buffer and offset in the flush buffer.
  static const size_t OffsetSize = 2 * sizeof(unsigned);
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
